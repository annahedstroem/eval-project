# %% [markdown]
# ## MNIST analysis
# 
# This notebook loads the MNIST dataset and a pretrained model and computes Qinv and Qbas for 100.000 random rankings.
# 
# :warning:
# # There's a .py version for this script in the src folder which is the one that should be used to generate the files:
# `src/extract_mnist_measures.py`

# %%
# Import the necessary libraries
import sys
import os
PROJ_DIR = os.path.realpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJ_DIR,'src'))
import xai_faithfulness_experiments_lib_edits as fl

import numpy as np

# %%
# MNIST dataset
DATASET = 'imagenet'
# Load dataset
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using {device}')
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import PIL
batch_size = 256
IMAGENETTE_PATH = os.path.join(PROJ_DIR, 'data', 'imagenette')
IMAGENETTE_CLASS_DICT = {'n01440764':0, 'n02102040':217, 'n02979186':481, 'n03000684':491, 'n03028079':497, 'n03394916':566, 'n03417042':569, 'n03425413':571, 'n03445777':574, 'n03888257':701}
IMAGENETTE_CLASS_DIRS = sorted(list(IMAGENETTE_CLASS_DICT.keys()))

def get_imagenette_dataset(is_test=False, project_path:str='../'):
    ''' Loads the imagenette dataset. By default it loads the train partition, unless otherwise indicated'''
    def transform_labels(l):
        new_l = IMAGENETTE_CLASS_DICT[IMAGENETTE_CLASS_DIRS[l]]
        return new_l

    def load_sample(path: str) -> dict:
        """Read data as image and path. """
        return PIL.Image.open(path).convert("RGB")


    #DATA_TRAIN_PATH = os.path.join(project_path, IMAGENETTE_PATH, 'train')
    #DATA_TEST_PATH = os.path.join(project_path, IMAGENETTE_PATH, 'val')
    # DEBUG!!!!
    DATA_TRAIN_PATH = os.path.join(project_path, IMAGENETTE_PATH, 'train_oneclass')
    DATA_TEST_PATH = os.path.join(project_path, IMAGENETTE_PATH, 'val_oneclass')

    transform = transforms.Compose([
                    transforms.Resize(256), 
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
    # Load test data and make loaders.
    dataset = torchvision.datasets.DatasetFolder(DATA_TEST_PATH if is_test else DATA_TRAIN_PATH, 
                                                loader=load_sample, 
                                                is_valid_file=lambda path: path[-5:]==".JPEG",
                                                transform=transform, # Should we do this here or work with the full images for the RL process??
                                                target_transform=transform_labels)
    return dataset

def get_imagenette_train_loader(batch_size:int = 24, project_path:str='../') -> torch.utils.data.DataLoader:
    dataset = get_imagenette_dataset(project_path=project_path)
    train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return train_loader

def get_imagenette_test_loader(batch_size:int = 24, project_path:str='../') -> torch.utils.data.DataLoader:
    dataset = get_imagenette_dataset(True, project_path=project_path)
    test_loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=batch_size)
    return test_loader

train_loader = get_imagenette_train_loader(52, PROJ_DIR)

examples = enumerate(train_loader)
batch_idx, (x_train, y_train) = next(examples)

MODEL_NAME = 'resnet50'
# Load model
network = torchvision.models.resnet50(weights="DEFAULT").to(device).eval()

from tqdm import tqdm
for GENERATOR in ['',  '_genetic']:
    num_rankings = 10000
    NUM_VARS  = 1
    INPUT_SHAPE = x_train.shape[1:]
    for d in x_train.shape[1:]:
        NUM_VARS *= d
    NUM_SAMPLES = min(fl.NUM_SAMPLES, NUM_VARS)

    # %%
    from tqdm import tqdm
    import quantus
    import genetic_generator as gg

    # The mean is zero because this dataset is standardized
    masking_values = torch.from_numpy(np.zeros(x_train.shape[1:])).float().to(device)

    for SAMPLE_NUM in [10, 20, 30, 40, 50]:
        print('Processing', SAMPLE_NUM)
        row = x_train[SAMPLE_NUM].clone().detach().to(device)
        label = y_train[SAMPLE_NUM].clone().detach().to(device)

        print(f'Generating {GENERATOR} rankings...')
        if GENERATOR == "_genetic":
            # Genetically optimized
            def fitness(ranking:np.ndarray) -> float:
                measures = fl.get_measures_for_ranking(row, torch.tensor(ranking, dtype=torch.float32).to(device), label, network, num_samples=NUM_SAMPLES, with_inverse=False, with_random=False, masking_values=masking_values)
                return measures['mean']
            all_rankings = gg.generate_rankings(num_rankings, INPUT_SHAPE, fitness, num_iterations = 50)
        else:
            #Random
            all_rankings = np.zeros((num_rankings, *INPUT_SHAPE)) # To be randomly generated on the first loop
            for i in range(num_rankings):
                all_rankings[i] = fl._get_random_ranking_row(row.shape).cpu().numpy() # Random generation

        # All of these measures will be stored
        suffixes = ['', '_inv']#, '_bas']
        size1_prefixes = ['mean', 'at_first_argmax']#, 'auc']
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
                all_measures[k+s] = np.zeros((num_rankings, NUM_SAMPLES), dtype=np.float32 if 'is_hit' not in k else bool)
        all_measures['ranking'] = np.zeros((num_rankings, *INPUT_SHAPE), dtype=np.float32)

        # Compute the results for each possible ranking
        for i in tqdm(range(num_rankings), miniters=1000):
            measures = fl.get_measures_for_ranking(row, torch.tensor(all_rankings[i], dtype=torch.float32).to(device), label, network, num_samples=NUM_SAMPLES, with_inverse=True, with_random=False, masking_values=masking_values)
            measures['ranking'] = all_rankings[i]
            # Save all results for this rankings to the i-th position
            for k in keys:
                all_measures[k][i] = measures[k]

        # %%
        #Retrieve and store Quantus' faithfulness metrics
        # To be used by Quantus
        x_batch_pt = torch.unsqueeze(row, dim=0)
        x_batch = x_batch_pt.to('cpu').numpy()
        y_batch = torch.unsqueeze(label, dim=0).to('cpu').numpy()

        all_measures['faithfulness_correlation'] = np.zeros(num_rankings, dtype=np.float32)
        #all_measures['monotonicity_correlation'] = np.zeros(num_rankings, dtype=np.float32)
        all_measures['pixel_flipping'] = np.zeros((num_rankings,NUM_VARS), dtype=np.float32)

        for i in tqdm(range(num_rankings),  miniters=1000):
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

        # %%
        np.savez(os.path.join(PROJ_DIR, 'results', f'{DATASET}_{SAMPLE_NUM}_{MODEL_NAME}{GENERATOR}_measures.npz'), \
                row=row.to('cpu').numpy(), \
                label=label.to('cpu').numpy(), \
                rankings=all_measures['ranking'], \
                faithfulness_correlations=all_measures['faithfulness_correlation'], \
                pixel_flippings=all_measures['pixel_flipping'], \
                qmeans=all_measures['mean'], \
                qmean_invs=all_measures['mean_inv'], \
                qargmaxs=all_measures['at_first_argmax'], \
                qargmax_invs=all_measures['at_first_argmax_inv'], \
                output_curves=all_measures['output_curve'], \
                is_hit_curves=all_measures['is_hit_curve'], \
                output_curves_inv=all_measures['output_curve_inv'], \
                is_hit_curves_inv=all_measures['is_hit_curve_inv'])#, \
                #inv_lookup=inv_lookup)
                #qmean_bas=all_measures['mean_bas'], \
                #qargmax_bas=all_measures['at_first_argmax_bas'], \
                #qauc_bas=all_measures['auc_bas'], \
                #output_curves_bas=all_measures['output_curve_bas'], \
                #is_hit_curves_bas=all_measures['is_hit_curve_bas']
                #monotonicity_correlations=all_measures['monotonicity_correlation'], \
                #qaucs=all_measures['auc'], \
                #qauc_invs=all_measures['auc_inv'], \

