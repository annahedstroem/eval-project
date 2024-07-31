# Import the necessary libraries
import sys
import os
PROJ_DIR = os.path.realpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJ_DIR,'src'))
import gce_lib as fl

import numpy as np
from tqdm import tqdm

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using {device}')

DEFAULT_CHUNKINESS = 32

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
            num_rankings = all_rankings.shape[0]
        elif generator_mode == "_chunky":
            #Random
            all_rankings = np.zeros((num_rankings, *input_shape)) # To be randomly generated on the first loop
            for i in tqdm(range(num_rankings)):
                all_rankings[i] = fl._get_chunky_random_ranking_row(row.shape, chunkiness, chunkiness, True).cpu().numpy() # Random generation
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
        size1_prefixes = ['mean']
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

        return all_measures

if __name__ == '__main__':
    BATCH_SIZE = 1000
    SAMPLE_INDICES = np.random.permutation(BATCH_SIZE).tolist()
    NEEDED_SAMPLES = 5
    NUM_RANKINGS = 10000

    #### Section 4.1.a
    tuples_to_test = [('avila', 'mlp', ['_full'], None), ('glass', 'mlp', ['_full'], None)]
    ##########
    
    #### Section 4.1.b - Table 1
    # tuples_to_test = [('20newsgroups', 'mlp', ['_random'], None), \
    #                   ('mnist', 'mlp', ['_random'], None), \
    #                   ('cifar', 'resnet50', ['_chunky'], 4), \
    #                   ('imagenet', 'resnet18', ['_chunky'], 32), \
    #                   ('imagenet', 'resnet50', ['_chunky'], 32), \
    #                   ('imagenet', 'vgg16', ['_chunky'], 32), \
    #                   ('imagenet', 'maxvit_t', ['_chunky'], 32), \
    #                   ('imagenet', 'vit_b_32', ['_chunky'], 32)]
    ##########

    #### Section 4.1.c - Table 2
    # tuples_to_test = [
    #     ('avila', 'ood-mean', ['_full'], None),
    #     ('avila', 'ood-zeros', ['_full'], None),
    #     ('avila', 'undertrained', ['_full'], None),
    #     ('avila', 'untrained', ['_full'], None),
    #     ('glass', 'ood-mean', ['_full'], None),
    #     ('glass', 'ood-zeros', ['_full'], None),
    #     ('glass', 'undertrained', ['_full'], None),
    #     ('glass', 'untrained', ['_full'], None),
    # ]
    ##########
    
    #### Appendix - Table 5
    # tuples_to_test = [('cifar', 'resnet50', ['_random'], None), \
    #                   ('imagenet', 'resnet18', ['_random'], None), \
    #                   ('imagenet', 'resnet50', ['_random'], None), \
    #                   ('imagenet', 'vgg16', ['_random'], None), \
    #                   ('imagenet', 'maxvit_t', ['_random'], None), \
    #                   ('imagenet', 'vit_b_32', ['_random'], None)]

    for DATASET, MODEL_NAME, GENERATORS, CHUNKINESS in tuples_to_test:
        # Load dataset
        if DATASET in ['20newsgroups', 'avila', 'glass']:
            DATASET_PATH = os.path.join(PROJ_DIR,'assets', 'data', f'{DATASET}.npz')
            # Load dataset
            file_data = np.load(DATASET_PATH)
            x_train = torch.from_numpy(file_data['x_train']).float().to(device)
            y_train = torch.from_numpy(file_data['y_train']).to(device)
        else:
            #torch.manual_seed(0)
            test_loader = fl.get_image_test_loader(DATASET, BATCH_SIZE, PROJ_DIR, shuffle = True)
        
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
        elif DATASET == '20newsgroups':
            MODEL_PATH = os.path.join(PROJ_DIR,'assets', 'models', f'{DATASET}-{MODEL_NAME}.pth')
            network = fl.load_pretrained_mlp_large_model(MODEL_PATH, x_train.shape[1], 20, [1000, 1000, 800, 500])
        elif DATASET == 'avila':
            MODEL_PATH = os.path.join(PROJ_DIR,'assets', 'models', f'{DATASET}-{MODEL_NAME}-mlp.pth')
            network = fl.load_pretrained_mlp_model(MODEL_PATH, 10, 12, 100)
        elif DATASET == 'glass':
            MODEL_PATH = os.path.join(PROJ_DIR,'assets', 'models', f'{DATASET}-{MODEL_NAME}-mlp.pth')
            network = fl.load_pretrained_mlp_model(MODEL_PATH, 9, 6, 100)
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

                all_measures = compute_measures_for_sample(network, row, label, masking_values, NUM_RANKINGS, num_samples, generator_name, chunkiness=CHUNKINESS)

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
