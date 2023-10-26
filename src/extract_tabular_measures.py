#!/usr/bin/env python
# coding: utf-8

# Import the necessary libraries
import sys
import os
PROJ_DIR = os.path.realpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJ_DIR,'src'))
import xai_faithfulness_experiments_lib_edits as fl

import numpy as np

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Compute the inv_lookup
#shape = tuple([NUM_VARS for x in range(NUM_VARS)])
#lookup = -1 * np.ones(shape, dtype=int)
#inv_lookup = -1 * np.ones(len(permutations), dtype=int)
#for i,p in tqdm(enumerate(permutations)):
#    lookup[p] = i
#    rev = tuple(reversed(p))
#    if lookup[rev] > 0:
#        inv_lookup[lookup[rev]] = i
#        inv_lookup[i] = lookup[rev]
#for i in range(0,len(inv_lookup)):
#    assert(permutations[i] == tuple(reversed(permutations[inv_lookup[i]])))

# Compute data for some samples
from tqdm import tqdm
import quantus

for DATASET in ['glass', 'avila']:
    print(f'Loading {DATASET} dataset...')
    DATASET_PATH = os.path.join(PROJ_DIR,'assets', 'data', f'{DATASET}.npz')
    # Load dataset
    file_data = np.load(DATASET_PATH)
    x_train = file_data['x_train']
    x_test = file_data['x_test']
    y_train = file_data['y_train']
    y_test = file_data['y_test']

    import itertools
    NUM_VARS = x_train.shape[1]
    permutations = list(itertools.permutations(range(NUM_VARS)))
    all_rankings = np.array(permutations) / (NUM_VARS - 1)
    NUM_SAMPLES = min(fl.NUM_SAMPLES, NUM_VARS)

    for MODEL_NAME in ['', '-ood-mean', '-ood-zeros', '-undertrained', '-untrained']:
        print(f'Loading {DATASET}{MODEL_NAME}-mlp model...')
        # Load model
        MODEL_NEURONS = 100
        MODEL_EPOCHS= 2000
        MODEL_LR = 1.0e-1
        MODEL_LABEL_NUM = len(np.unique(y_train))

        
        MODEL_PATH = os.path.join(PROJ_DIR,'assets', 'models', f'{DATASET}{MODEL_NAME}-mlp.pth')
        network = fl.load_pretrained_mlp_model(MODEL_PATH, x_train.shape[1], MODEL_LABEL_NUM, MODEL_NEURONS)

        for SAMPLE_NUM in [10, 20, 30, 40, 50]:#, 80, 90, 100, 150, 300]:
            print('Processing', SAMPLE_NUM)

            masking_values = torch.from_numpy(np.mean(x_train, axis=0)).float().to(device) if MODEL_NAME == '-ood-mean' else torch.from_numpy(np.zeros(x_train.shape[1])).float().to(device)

            num_rankings = all_rankings.shape[0]
            row = torch.tensor(np.float32(x_train[SAMPLE_NUM])).to(device)
            label = torch.tensor(y_train[SAMPLE_NUM]).to(device)

            # All of these measures will be stored
            suffixes = ['', '_inv', '_bas']
            size1_prefixes = ['mean', 'at_first_argmax', 'auc']
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
            all_measures['ranking'] = np.zeros((num_rankings, NUM_VARS), dtype=np.float32)

            # Compute the results for each possible ranking
            for i in tqdm(range(num_rankings), miniters=10000):
                measures = fl.get_measures_for_ranking(row, torch.tensor(all_rankings[i], dtype=torch.float32).to(device), label, network, num_samples=NUM_SAMPLES, with_inverse=True, with_random=True, masking_values=masking_values)
                measures['ranking'] = all_rankings[i]
                # Save all results for this rankings to the i-th position
                for k in keys:
                    all_measures[k][i] = measures[k]


            #Retrieve and store Quantus' faithfulness metrics
            # To be used by Quantus
            x_batch_pt = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(row, dim=0), dim=0), dim=0)
            x_batch = x_batch_pt.to('cpu').numpy()
            y_batch = torch.unsqueeze(label, dim=0).to('cpu').numpy()

            all_measures['faithfulness_correlation'] = np.zeros(num_rankings, dtype=np.float32)
            all_measures['monotonicity_correlation'] = np.zeros(num_rankings, dtype=np.float32)
            all_measures['pixel_flipping'] = np.zeros((num_rankings,NUM_VARS), dtype=np.float32)

            for i in tqdm(range(num_rankings),  miniters=10000):
                #For each ranking, retrieve and store Quantus' faithfulness metrics
                a_batch = np.expand_dims(np.expand_dims(np.expand_dims(all_rankings[i],0),0),0)
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
                all_measures['monotonicity_correlation'][i] = quantus.MonotonicityCorrelation(
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

            np.savez(os.path.join(PROJ_DIR, 'results', f'{DATASET}_{SAMPLE_NUM}{MODEL_NAME}_measures.npz'), \
                    row=row.to('cpu').numpy(), \
                    label=label.to('cpu').numpy(), \
                    rankings=all_measures['ranking'], \
                    faithfulness_correlations=all_measures['faithfulness_correlation'], \
                    monotonicity_correlations=all_measures['monotonicity_correlation'], \
                    pixel_flippings=all_measures['pixel_flipping'], \
                    qmeans=all_measures['mean'], \
                    qmean_invs=all_measures['mean_inv'], \
                    qmean_bas=all_measures['mean_bas'], \
                    qargmaxs=all_measures['at_first_argmax'], \
                    qargmax_invs=all_measures['at_first_argmax_inv'], \
                    qargmax_bas=all_measures['at_first_argmax_bas'], \
                    qaucs=all_measures['auc'], \
                    qauc_invs=all_measures['auc_inv'], \
                    qauc_bas=all_measures['auc_bas'], \
                    output_curves=all_measures['output_curve'], \
                    is_hit_curves=all_measures['is_hit_curve'], \
                    output_curves_inv=all_measures['output_curve_inv'], \
                    is_hit_curves_inv=all_measures['is_hit_curve_inv'], \
                    output_curves_bas=all_measures['output_curve_bas'], \
                    is_hit_curves_bas=all_measures['is_hit_curve_bas'])#, \
                    #inv_lookup=inv_lookup)