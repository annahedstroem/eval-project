# Import the necessary libraries
import sys
import os
PROJ_DIR = os.path.realpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJ_DIR,'src'))
import gce_lib as fl

import numpy as np
from tqdm import tqdm
import quantus
from quantus.metrics.faithfulness import FaithfulnessCorrelation, FaithfulnessEstimate, MonotonicityCorrelation, Sufficiency

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using {device}')

DEFAULT_CHUNKINESS = 16

quantus_functions = {'FaithfulnessCorrelation': FaithfulnessCorrelation(
                                                            nr_runs=10,
                                                            subset_size=224,  
                                                            perturb_baseline="black",
                                                            perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                                                            similarity_func=quantus.similarity_func.correlation_pearson,  
                                                            abs=False,
                                                            normalise=False,
                                                            return_aggregate=False,
                                                            disable_warnings=True
                                                        ),\
                    'FaithfulnessEstimate': FaithfulnessEstimate(
                                                            perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                                                            similarity_func=quantus.similarity_func.correlation_pearson,
                                                            features_in_step=224,  
                                                            perturb_baseline="black",
                                                            disable_warnings=True
                                                        ),
                    'MonotonicityCorrelation': MonotonicityCorrelation(
                                            nr_samples=10,
                                            features_in_step=3136,
                                            perturb_baseline="uniform",
                                            perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                                            similarity_func=quantus.similarity_func.correlation_spearman,
                                            disable_warnings=True
                                        ),
                    'Sufficiency' : Sufficiency(
                                            threshold=0.6,
                                            return_aggregate=False,
                                            disable_warnings=True
                                        )
        }

def compute_measures_for_sample(network:torch.nn.Module,\
                                row:torch.Tensor,\
                                label:torch.Tensor,\
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
        else:
            #Random
            all_rankings = np.zeros((num_rankings, *input_shape)) # To be randomly generated on the first loop
            for i in tqdm(range(num_rankings)):
                all_rankings[i] = fl._get_random_ranking_row(row.shape).cpu().numpy() # Random generation

        # Dict to store all results
        all_measures = {}

        all_measures['ranking'] = np.zeros((num_rankings, *input_shape), dtype=np.float32)
        print(f'Computing measures...')

        #Retrieve and store Quantus' faithfulness metrics
        # To be used by Quantus
        x_batch_pt = torch.unsqueeze(row, dim=0)
        x_batch = x_batch_pt.to('cpu').numpy()
        y_batch = torch.unsqueeze(label, dim=0).to('cpu').numpy()

        for k in quantus_functions:
            all_measures[k] = np.zeros(num_rankings, dtype=np.float32)
            all_measures[k + '_inv'] = np.zeros(num_rankings, dtype=np.float32)

        for i in tqdm(range(num_rankings),  miniters=1000):
            all_measures['ranking'][i] = all_rankings[i]
            #For each ranking, retrieve and store Quantus' faithfulness metrics
            a_batch = np.expand_dims(all_rankings[i], 0)
            a_batch_bas = np.expand_dims(fl._get_random_ranking_row(row.shape).cpu().numpy(), 0)
            a_batch_inv = np.expand_dims(1-all_rankings[i], 0)

            for k in quantus_functions:
                f = quantus_functions[k]

                all_measures[k][i] = f(model=network,
                                        x_batch=x_batch, 
                                        y_batch=y_batch,
                                        a_batch=a_batch,
                                        device=device,
                                        channel_first=True)[0]
                all_measures[k + '_inv'][i] = all_measures[k][i] - f(model=network, 
                                                x_batch=x_batch, 
                                                y_batch=y_batch,
                                                a_batch=a_batch_inv,
                                                device=device,
                                                channel_first=True)[0]
        return all_measures

if __name__ == '__main__':
    NUM_RANKINGS = 1000
    NEEDED_SAMPLES = 5
    SAMPLE_INDICES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 1, 11, 21, 31, 41, 51, 61, 71, 81, 91]

    #### Section 4.1.d - Table 3
    tuples_to_test = [
                      ('imagenet', 'resnet18', ['_chunky']), \
                      ('imagenet', 'vgg16', ['_chunky']), \
    ]
    ##########

    for DATASET, MODEL_NAME, GENERATORS in tuples_to_test:    
        # Load dataset
        if DATASET in ['avila', 'glass', '20newsgroups-truncated']:
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
        elif DATASET in ['avila']:
            MODEL_PATH = os.path.join(PROJ_DIR,'assets', 'models', f'{DATASET}{MODEL_NAME}-mlp.pth')
            network = fl.load_pretrained_mlp_model(MODEL_PATH, x_train.shape[1], 12, 100)
        elif DATASET in ['glass']:
            MODEL_PATH = os.path.join(PROJ_DIR,'assets', 'models', f'{DATASET}{MODEL_NAME}-mlp.pth')
            network = fl.load_pretrained_mlp_model(MODEL_PATH, x_train.shape[1], 6, 100)
        else:
            raise Exception(f'ERROR: Unknown dataset {DATASET}')

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
                
                print(f'Processing {sample_index} ({network(row.unsqueeze(dim=0))[0,label].item()})')
                if network(row.unsqueeze(dim=0))[0,label].item() < 0.75:
                    #raise Exception(f'Sample {sample_index} does not have high activation')
                    print(f'Skipping sample {sample_index} for not having high activation')
                    continue
                
                correct_samples += 1

                with torch.no_grad():
                    all_measures = compute_measures_for_sample(network, row, label, masking_values, NUM_RANKINGS, num_samples, generator_name)

                quantus_results = {}
                for k in quantus_functions:
                    quantus_results[k] = all_measures[k]
                    quantus_results[k + '_inv'] = all_measures[k + '_inv']

                np.savez(os.path.join(PROJ_DIR, 'results', f'{DATASET}_{sample_index}_{MODEL_NAME}{generator_name}_quantus_other_measures.npz'), \
                        row=row.to('cpu').numpy(), \
                        label=label.to('cpu').numpy(), \
                        rankings=all_measures['ranking'], \
                        **quantus_results)
