# Import the necessary libraries
import sys
import os
PROJ_DIR = os.path.realpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJ_DIR,'src'))
import gce_lib as fl

import numpy as np
from tqdm import tqdm
from quantus import AttributionLocalisation, TopKIntersection, RelevanceRankAccuracy, RelevanceMassAccuracy, AUC

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using {device}')

DEFAULT_CHUNKINESS = 4

def manual_attribution_localization(model, x_batch, y_batch, a_batch, s_batch):
    '''Assumes single element batches'''
    # Prepare shapes.
    a = a_batch.flatten()
    a = np.clip(a, 0, None)
    s = s_batch.flatten().astype(bool)

    # Compute ratio.
    size_bbox = float(np.sum(s))
    size_data = s.size
    ratio = size_bbox / size_data

    # Compute inside/outside ratio.
    inside_attribution = np.sum(a[s])
    total_attribution = np.sum(a)
    inside_attribution_ratio = float(inside_attribution / total_attribution)
    return np.array([inside_attribution_ratio])

localization_functions = {'AttributionLocalisation': manual_attribution_localization,\
                          'TopKIntersection': TopKIntersection(disable_warnings=True),\
                          'RelevanceRankAccuracy': RelevanceRankAccuracy(disable_warnings=True),\
                          'RelevanceMassAccuracy': RelevanceMassAccuracy(disable_warnings=True),\
                          'AUC': AUC(disable_warnings=True)}
#AttributionLocalisation(disable_warnings=True),\

def compute_measures_for_sample(network:torch.nn.Module,\
                                row:torch.Tensor,\
                                label:torch.Tensor,\
                                s_mask:torch.Tensor,\
                                masking_values:torch.Tensor,\
                                num_rankings: int,\
                                num_samples:int = 20,\
                                generator_mode:str = '',\
                                with_inverse:bool = True,\
                                with_random:bool = False,\
                                chunkiness:int = DEFAULT_CHUNKINESS) -> dict:
        input_shape = row.shape
        print(f'Generating {generator_mode} rankings...')
        if generator_mode == "_full":
            import itertools
            num_vars = 1
            for s in input_shape:
                num_vars *= s
            permutations = list(itertools.permutations(range(num_vars)))
            all_rankings = np.reshape(np.array(permutations) / (num_vars - 1), (len(permutations), *input_shape))
        elif generator_mode == "_chunky":
            #Random
            all_rankings = np.zeros((num_rankings, *input_shape)) # To be randomly generated on the next loop
            for i in tqdm(range(num_rankings)):
                all_rankings[i] = fl._get_chunky_random_ranking_row(row.shape, chunkiness, chunkiness, True).cpu().numpy() # Random generation
        elif generator_mode == "_randomattr":
            #Random attribution
            all_rankings = np.random.normal(size=(num_rankings, *input_shape))# Random attributions
        else:
            #Random
            all_rankings = np.zeros((num_rankings, *input_shape)) # To be randomly generated on the next loop
            for i in tqdm(range(num_rankings)):
                all_rankings[i] = fl._get_random_ranking_row(row.shape).cpu().numpy() # Random generation

        # Dict to store all results
        all_measures = {}

        all_measures['ranking'] = np.zeros((num_rankings, *input_shape), dtype=np.float32)
        print(f'Computing measures...')

        #Retrieve and store Quantus' localization metrics
        # To be used by Quantus
        x_batch_pt = torch.unsqueeze(row, dim=0)
        x_batch = x_batch_pt.to('cpu').numpy()
        y_batch = torch.unsqueeze(label, dim=0).to('cpu').numpy()
        s_batch = np.expand_dims(s_mask.detach().cpu().numpy(), axis=0)

        for k in localization_functions:
            all_measures[k] = np.zeros(num_rankings, dtype=np.float32)
            all_measures[k + '_inv'] = np.zeros(num_rankings, dtype=np.float32)

        for i in tqdm(range(num_rankings),  miniters=1000):
            all_measures['ranking'][i] = all_rankings[i]
            #For each ranking, retrieve and store Quantus' localization metrics
            ranking = all_rankings[i]
            a_batch = np.expand_dims(ranking.sum(axis=0, keepdims=True), 0)
            inverse = 1-ranking
            if generator_mode=='_randomattr':
                inverse = fl.get_inverse(ranking)
            a_batch_inv = np.expand_dims(inverse.sum(axis=0, keepdims=True), 0)
            for k in localization_functions:
                f = localization_functions[k]
                all_measures[k][i] = f(model=network, 
                                        x_batch=x_batch, 
                                        y_batch=y_batch,
                                        a_batch=a_batch,
                                        s_batch=s_batch)[0]
                all_measures[k + '_inv'][i] = all_measures[k][i] - f(model=network, 
                                                                    x_batch=x_batch, 
                                                                    y_batch=y_batch,
                                                                    a_batch=a_batch_inv,
                                                                    s_batch=s_batch)[0]
        return all_measures

if __name__ == '__main__':
    NUM_RANKINGS = 10000
    NEEDED_SAMPLES = 5
    SAMPLE_INDICES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 1, 11, 21, 31, 41, 51, 61, 71, 81, 91]

    #### Section 4.1.d - Table 4
    tuples_to_test = [('cmnist', 'resnet18', ['_chunky'])]
    ##########

    for DATASET, MODEL_NAME, GENERATORS in tuples_to_test:
        # Load dataset
        test_set = fl.CMNISTDataset(dict_file_path=os.path.join(PROJ_DIR, 'data', 'cmnist_test_dict.pickle'))
        mask_name = 's_area'
        x_train = torch.tensor(list(map(lambda x: x['x'], test_set.data.values())))
        y_train = torch.tensor(list(map(lambda x: x['y'], test_set.data.values())))
        s_train = torch.tensor(list(map(lambda x: x[mask_name], test_set.data.values())))

        s_train = s_train[:,0,:,:].unsqueeze(1) # Only keep first channel of the masks

        # Load model
        network = fl.load_pretrained_cmnist_resnet18_model(os.path.join(PROJ_DIR,'assets','models','cmnist-resnet18.pth'))

        num_vars  = 1
        for d in x_train.shape[1:]:
            num_vars *= d
        num_samples = min(fl.NUM_SAMPLES, num_vars)

        # The mean is zero because this dataset is standardized
        masking_values = torch.from_numpy(np.zeros(x_train.shape[1:])).float().to(device)

        for generator_name in GENERATORS:
            correct_samples = 0
            for sample_index in SAMPLE_INDICES:
                if correct_samples >= NEEDED_SAMPLES:
                    break
                row = x_train[sample_index].clone().detach().to(device)
                label = y_train[sample_index].clone().detach().to(device)
                s_mask = s_train[sample_index].clone().detach().to(device)
                
                softmax_output = torch.nn.functional.softmax(network(row.unsqueeze(dim=0)), dim=1)[0,label].item()

                print(f'Processing {sample_index} ({softmax_output})')
                if softmax_output < 0.75:
                    #raise Exception(f'Sample {sample_index} does not have high activation')
                    print(f'Skipping sample {sample_index} for not having high activation')
                    continue
                
                correct_samples += 1

                with torch.no_grad():
                    all_measures = compute_measures_for_sample(network, row, label, s_mask, masking_values, NUM_RANKINGS, num_samples, generator_name, chunkiness=DEFAULT_CHUNKINESS)

                localization_results = {}
                for k in localization_functions:
                    localization_results[k] = all_measures[k]
                    localization_results[k + '_inv'] = all_measures[k + '_inv']
                
                np.savez(os.path.join(PROJ_DIR, 'results', f'{DATASET}_{sample_index}_{MODEL_NAME}{generator_name}_localization_{mask_name}_measures.npz'), \
                        row=row.to('cpu').numpy(), \
                        s_mask=s_mask.to('cpu').numpy(), \
                        label=label.to('cpu').numpy(), \
                        rankings=all_measures['ranking'], \
                        **localization_results)
