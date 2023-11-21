# Import the necessary libraries
import sys
import os
PROJ_DIR = os.path.realpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJ_DIR,'src'))
import xai_faithfulness_experiments_lib_edits as fl

import numpy as np
from tqdm import tqdm
import quantus
import genetic_generator as gg
import captum_generator as cg

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using {device}')

def compute_measures_for_sample(row:torch.Tensor,\
                                label:torch.Tensor,\
                                masking_values:torch.Tensor,\
                                num_rankings: int,\
                                num_samples:int = 20,\
                                generator_mode:str = '',\
                                with_inverse:bool = True,\
                                with_random:bool = False,\
                                genetic_iterations: int = 50,\
                                with_quantus:bool = False) -> dict:
        input_shape = row.shape
        print(f'Generating {generator_mode} rankings...')
        if generator_mode == "_full":
            import itertools
            num_vars = 1
            for s in input_shape:
                num_vars *= s
            permutations = list(itertools.permutations(range(num_vars)))
            all_rankings = np.reshape(np.array(permutations) / (num_vars - 1), (len(permutations), *input_shape))
        elif generator_mode == "_genetic":
            # Genetically optimized
            def fitness(ranking:np.ndarray) -> float:
                measures = fl.get_measures_for_ranking(row, torch.tensor(ranking, dtype=torch.float32).to(device), label, network, num_samples=num_samples, with_inverse=with_inverse, with_random=with_random, masking_values=masking_values)
                return measures['mean']
            all_rankings = gg.generate_rankings(num_rankings, input_shape, fitness, num_iterations = genetic_iterations)
        elif generator_mode == "_captum":
            all_rankings = cg.generate_rankings(row, label, network)
            num_rankings = all_rankings.shape[0]
        else:
            #Random
            all_rankings = np.zeros((num_rankings, *input_shape)) # To be randomly generated on the first loop
            for i in tqdm(range(num_rankings)):
                all_rankings[i] = fl._get_random_ranking_row(row.shape).cpu().numpy() # Random generation

        # All of these measures will be stored
        suffixes = ['']
        if with_inverse:
            suffixes.append('_inv')
        if with_random:
            suffixes.append('_bas')
        size1_prefixes = ['mean']#, 'at_first_argmax'], 'auc']
        sizeNUM_SAMPLES_prefixes = ['output_curve', 'is_hit_curve']
        keys = ['ranking']
        for p in size1_prefixes+sizeNUM_SAMPLES_prefixes:
            for s in suffixes:
                keys.append(p+s)

        # Dict to store all results
        all_measures = {}
        # Initialize all np arrays to speed up the process
        for k in size1_prefixes:
            for s in suffixes:
                all_measures[k+s] = np.zeros((num_rankings, 1), dtype=np.float32)

        for k in sizeNUM_SAMPLES_prefixes:
            for s in suffixes:
                all_measures[k+s] = np.zeros((num_rankings, num_samples), dtype=np.float32 if 'is_hit' not in k else bool)
        all_measures['ranking'] = np.zeros((num_rankings, *input_shape), dtype=np.float32)

        print(f'Computing measures...')
        # Compute the results for each possible ranking
        for i in tqdm(range(num_rankings), miniters=1000):
            measures = fl.get_measures_for_ranking(row, torch.tensor(all_rankings[i], dtype=torch.float32).to(device), label, network, num_samples=num_samples, with_inverse=True, with_random=False, masking_values=masking_values)
            measures['ranking'] = all_rankings[i]
            # Save all results for this rankings to the i-th position
            for k in keys:
                all_measures[k][i] = measures[k]

        if with_quantus:
            #Retrieve and store Quantus' faithfulness metrics
            # To be used by Quantus
            x_batch_pt = torch.unsqueeze(row, dim=0)
            x_batch = x_batch_pt.to('cpu').numpy()
            y_batch = torch.unsqueeze(label, dim=0).to('cpu').numpy()

            all_measures['faithfulness_correlation'] = np.zeros(num_rankings, dtype=np.float32)
            #all_measures['monotonicity_correlation'] = np.zeros(num_rankings, dtype=np.float32)
            all_measures['pixel_flipping'] = np.zeros((num_rankings,num_vars), dtype=np.float32)

            for i in []:# tqdm(range(num_rankings),  miniters=1000):
                #For each ranking, retrieve and store Quantus' faithfulness metrics
                a_batch = np.expand_dims(all_rankings[i], 0)
                #print('x_batch shape:',x_batch.shape)
                #print('y_batch shape:',y_batch.shape)
                #print('a_batch shape:',a_batch.shape)
                #print('network(x_batch) shape:',network(torch.tensor(x_batch).to(device)).shape)
                #print(a_batch)
                all_measures['faithfulness_correlation'][i] = quantus.FaithfulnessCorrelation(
                                                                nr_runs=10,
                                                                subset_size=4,  
                                                                perturb_baseline="black",
                                                                perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                                                                similarity_func=quantus.similarity_func.correlation_pearson,  
                                                                abs=False,  
                                                                return_aggregate=False,
                                                                disable_warnings=True
                                                            )(model=network, 
                                                            x_batch=x_batch, 
                                                            y_batch=y_batch,
                                                            a_batch=a_batch,
                                                            device=device,
                                                            channel_first=True)[0]
                '''all_measures['monotonicity_correlation'][i] = quantus.MonotonicityCorrelation(
                                                                nr_samples=10,
                                                                features_in_step=2 if NUM_VARS % 2 == 0 else 1,
                                                                perturb_baseline="black",
                                                                perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                                                                similarity_func=quantus.similarity_func.correlation_spearman,
                                                                disable_warnings=True
                                                            )(model=network, 
                                                            x_batch=x_batch,
                                                            y_batch=y_batch,
                                                            a_batch=a_batch,
                                                            device=device,
                                                            channel_first=True)[0]
                '''
                all_measures['pixel_flipping'][i] = quantus.PixelFlipping(
                                                                features_in_step=1,
                                                                perturb_baseline="black",
                                                                perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                                                                disable_warnings=True
                                                            )(model=network,
                                                                x_batch=x_batch,
                                                                y_batch=y_batch,
                                                                a_batch=a_batch,
                                                                device=device,
                                                            channel_first=True)[0]
        return all_measures

if __name__ == '__main__':
    DATASET = 'imagenet'
    MODEL_NAME = 'vgg16'
    GENERATORS = ['', '_genetic','_captum']
    #SAMPLE_INDICES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 1, 11, 21, 31, 41, 51, 61, 71, 81, 91]
    SAMPLE_INDICES = [2, 12, 22, 32, 42]
    #SAMPLE_INDICES = [10, 20, 30, 40, 50]
    NUM_RANKINGS = 10000
    GENETIC_ITERATIONS = 50

    # Load dataset
    if DATASET == '20newsgroups-truncated':
        DATASET_PATH = os.path.join(PROJ_DIR,'assets', 'data', f'{DATASET}.npz')
        # Load dataset
        file_data = np.load(DATASET_PATH)
        x_train = torch.from_numpy(file_data['x_train']).float().to(device)
        y_train = torch.from_numpy(file_data['y_train']).to(device)
    else:
        #torch.manual_seed(0)
        test_loader = fl.get_image_test_loader(DATASET, 100, PROJ_DIR, shuffle = True)
    
        examples = enumerate(test_loader)
        batch_idx, (x_train, y_train) = next(examples)


    # Load model
    if DATASET == 'imagenet':
        network = fl.load_pretrained_imagenet_model(arch = MODEL_NAME)
    elif DATASET == 'mnist':
        MODEL_PATH = os.path.join(PROJ_DIR,'assets', 'models', f'{DATASET}-{MODEL_NAME}-mlp.pth')
        network = fl.load_pretrained_mnist_model(MODEL_PATH)
    elif DATASET == 'cifar':
        MODEL_PATH = os.path.join(PROJ_DIR,'assets', 'models', f'{DATASET}-{MODEL_NAME}-mlp.pth')
        network = fl.load_pretrained_cifar_model(MODEL_PATH)
    elif DATASET == '20newsgroups-truncated':
        MODEL_PATH = os.path.join(PROJ_DIR,'assets', 'models', f'{DATASET}{MODEL_NAME}-mlp.pth')
        network = fl.load_pretrained_mlp_large_model(MODEL_PATH, x_train.shape[1], 20, [1000, 1000, 800, 500])
    else:
        raise Exception(f'ERROR: Unknown dataset {DATASET}')

    num_vars  = 1
    for d in x_train.shape[1:]:
        num_vars *= d
    num_samples = min(fl.NUM_SAMPLES, num_vars)

    # The mean is zero because this dataset is standardized
    masking_values = torch.from_numpy(np.zeros(x_train.shape[1:])).float().to(device)

    for generator_name in GENERATORS:
        for sample_index in SAMPLE_INDICES:
            print('Processing', sample_index)
            row = x_train[sample_index].clone().detach().to(device)
            label = y_train[sample_index].clone().detach().to(device)
            
            all_measures = compute_measures_for_sample(row, label, masking_values, NUM_RANKINGS, num_samples, generator_name, genetic_iterations=GENETIC_ITERATIONS)

            np.savez(os.path.join(PROJ_DIR, 'results', f'{DATASET}_{sample_index}_{MODEL_NAME}{generator_name}_measures.npz'), \
                    row=row.to('cpu').numpy(), \
                    label=label.to('cpu').numpy(), \
                    rankings=all_measures['ranking'], \
                    qmeans=all_measures['mean'], \
                    qmean_invs=all_measures['mean_inv'], \
                    output_curves=all_measures['output_curve'], \
                    is_hit_curves=all_measures['is_hit_curve'], \
                    output_curves_inv=all_measures['output_curve_inv'], \
                    is_hit_curves_inv=all_measures['is_hit_curve_inv'])
