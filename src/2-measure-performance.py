# %% [markdown]
# # Measure performance
# This notebook loads a file with precomputed measures (*qmeans*, *qbas* & *qinv*) for a set of rankings for a given instance of the dataset and measures the performance of the different alternative measures
# 
# ## 1. Load libraries, model and data

# %%

# Import the necessary libraries
import sys
import os
PROJ_DIR = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(PROJ_DIR,'src'))
import xai_faithfulness_experiments_lib_edits as fl
import numpy as np
from typing import Optional
from matplotlib import pyplot as plt

DATASET = 'cmnist'
MODEL_NAME = 'resnet18'
GENERATIONS = ['_random', '_captum', '_chunky']
TARGET_MEASURES = ['AttributionLocalisation', 'TopKIntersection', 'RelevanceRankAccuracy', 'AUC'] # 'qmeans' | 'faithfulness_correlation' | 'AttributionLocalisation' | 'TopKIntersection' | 'RelevanceRankAccuracy' | 'AUC'

for TARGET_MEASURE in TARGET_MEASURES:
    for GENERATION in GENERATIONS:
        for FILENAME in os.listdir(os.path.join(PROJ_DIR,'results')):
            if FILENAME.startswith(DATASET) and FILENAME.endswith(f'{MODEL_NAME}{GENERATION}_localization_s_area_measures.npz'):
                print(FILENAME)

                # Load data
                data = fl.load_generated_data(os.path.join(PROJ_DIR, 'results', FILENAME))

                for k in data.keys():
                    print(k)
                
                qmeans = data[TARGET_MEASURE]
                #qmeans_basX = [data['qmean_bas']] # We don't look at qmean_bas, it will be recomputed later with the appropriate reference
                qmeans_basX = []
                qmeans_inv = data['qmean_invs' if TARGET_MEASURE=='qmeans' else TARGET_MEASURE + '_inv']

                # Compute qmeans_bas[2-10]
                def compute_qbas(measure, num_samples, reference:np.ndarray):
                    random_indices = np.random.randint(0, measure.shape[0], (measure.shape[0], num_samples))
                    random_qmeans = reference[random_indices]
                    mean = np.mean(random_qmeans, axis=1)

                    # First way to deal with std==0; add some epsilon
                    #std = np.std(random_qmeans, axis=1) + 1e-10

                    # Second way to deal with std==0; ignore std (divide by 1)
                    std = np.std(random_qmeans, axis=1)
                    std[std==0] = 1

                    # Always ignore std
                    std=1
                    return (measure - mean) / std
                
                if GENERATION in ['_genetic', '_captum']:
                    # If data is genetic, we'll load the random generated equivalent to compute qbas with
                    data_reference = fl.load_generated_data(os.path.join(PROJ_DIR, 'results', FILENAME.replace(GENERATION, '_random'))) # or '_random'
                    qmeans_reference = data_reference[TARGET_MEASURE]

                for i in range(1,11):
                    # If data is genetic, compute qbas with random data from other file
                    qmeans_basX.append(compute_qbas(qmeans, i, qmeans_reference if GENERATION in ['_genetic', '_captum'] else qmeans))

                # Compute z-score
                qmean_mean = np.mean(qmeans)
                qmean_std = np.std(qmeans)
                
                if GENERATION in ['_genetic', '_captum']:
                    qmean_mean = np.mean(qmeans_reference)
                    qmean_std = np.std(qmeans_reference)

                z_scores = ((qmeans - qmean_mean) / qmean_std).flatten()

                # Stratify z-index to be able to compare performance on different parts of the spectrum
                indices = np.arange(z_scores.shape[0])
                z_scores_numbered = np.vstack((z_scores, indices))
                level_indices = []
                boundaries = [0.5, 1, 1.5, 2, 2.5]#, 3, 3.5]
                for i in range(1,len(boundaries)+1):
                    bottom_limit = boundaries[i-1]
                    top_limit = float('inf')
                    if i < len(boundaries):
                        top_limit = boundaries[i]
                    level_indices.append((z_scores_numbered[:,np.logical_and(bottom_limit<=z_scores, z_scores<top_limit)][1,:].astype(int),(bottom_limit, top_limit)))
                
                # %% [markdown]
                # ## 2. Measure performance
                # ### 2.1 Order preservation
                #  1. The issue with using qmean directly is that it doesn't have a fixed scale and you don't get an idea of how good your explanation is compared to other explanations
                #  2. To address this, ideally you would determine the distribution of all qmeans and then compute the z-score. That's very costly, so you either:
                #     1. Estimate the qmeans distribution with X samples $\rightarrow$ qbasX
                #     2. Calculate an alternative to the z-index directly $\rightarrow$ qinv
                #  3. The problem with both alternatives is that you adulterate the value of your original qmean measurement, so you may end up in a situation where $qmean_i<qmean_j$ but $qinv_i<qinv_j$, which is undesirable
                #  4. Hence, we measure how many times that happens for each measure.
                # 
                #  (This may be measuring the same as Pearson correlation, which is computed below)

                # %%
                def measure_correct_orderings(truths, estimators):
                    '''
                    Creates len(truth) x,y pairs and computes the fraction of them for which (truths[x]<truths[y] and estimators[x]<estimators[y]) or (truths[x]>truths[y] and estimators[x]>estimators[y])
                    Inputs:
                        - Truths & estimators contain num_elems floats
                    Output:
                        - Float representing the fraction of correctly ordered pairings
                    '''
                    xs = np.random.permutation(truths.size)
                    ys = np.random.permutation(truths.size)
                    truthX_lt_Y = truths[xs] < truths[ys]
                    estimatorX_lt_Y = estimators[xs] < estimators[ys]
                    hits = truthX_lt_Y==estimatorX_lt_Y
                    return hits.sum()/truths.size

                print('\tCorrect orderings:')
                correct_pairings_basX = []
                for i in range(len(qmeans_basX)):
                    #correct_pairings_basX.append(measure_correct_orderings(qmeans, qmeans_basX[i]))
                    correct_pairings_basX.append(measure_correct_orderings(z_scores, qmeans_basX[i]))
                    print(f'\t\tqmeans_bas{i+1}: {correct_pairings_basX[i]:.4f}')
                #correct_pairings_inv = measure_correct_orderings(qmeans, qmeans_inv)
                correct_pairings_inv = measure_correct_orderings(z_scores, qmeans_inv)#TODO Check in paper what shoud be compared
                print('\t\t'+'-'*20)
                print(f'\t\tqmeans_inv: {correct_pairings_inv:.4f}')

                # %% [markdown]
                # ### 2.2. Spearman correlation
                # Same thing, is the order of qmeans preserved in qbasX/qinv?

                # %%
                print('\tSpearman correlation:')
                from scipy.stats import spearmanr
                spearman_basX = []
                for i in range(len(qmeans_basX)):
                    spearman_basX.append(spearmanr(qmeans, qmeans_basX[i])[0])
                    print(f'\t\tqmeans_bas{i+1}: {spearman_basX[i]:.4f}')
                spearman_inv = spearmanr(qmeans, qmeans_inv)[0]
                print('\t\t'+'-'*20)
                print(f'\t\tqmeans_inv: {spearman_inv:.4f}')

                # %% [markdown]
                # ### 2.3. Ability to detect exceptionally good rankings
                # As stated above, there are some ordering errors in the estimators. Are they in the relevant part of the distribution? i.e. Do they affect the ability to identify exceptionally good rankings?

                # %%
                from sklearn import metrics

                def measure_detection(target_indices, estimator):
                    if (len(target_indices)==0) or (len(target_indices) == estimator.shape[0]):
                        return float('nan')
                    target = np.zeros_like(estimator, dtype=int)
                    target[target_indices] = 1
                    return metrics.roc_auc_score(target, estimator)

                aucs_inv = []
                aucs_basX = [[] for i in qmeans_basX]

                for indices, (bottom_limit, upper_limit) in level_indices:
                    aucs_inv.append(measure_detection(indices, qmeans_inv))
                    for i in range(len(qmeans_basX)):
                        aucs_basX[i].append(measure_detection(indices, qmeans_basX[i]))

                print('\tExceptional detection:')
                for i in range(len(qmeans_basX)):
                    print(f'\t\taucs_bas{i} ' + ' | '.join(map(lambda x: f'{x:.4f}',aucs_basX[i])))
                print('\t\t'+'-'*20)
                print('\t\taucs_inv ' + ' | '.join(map(lambda x: f'{x:.4f}',aucs_inv)))


                # %% [markdown]
                # ### 2.4 Ability to rank exceptionally good rankings
                # How well is the order preserved for exceptionally good rankings?

                # %%
                spearman_exceptional_inv = []
                spearman_exceptional_basX = [[] for i in qmeans_basX]

                for indices, (bottom_limit, upper_limit) in level_indices:
                    spearman_exceptional_inv.append(spearmanr(qmeans[indices], qmeans_inv[indices])[0])
                    for i in range(len(qmeans_basX)):
                        spearman_exceptional_basX[i].append(spearmanr(qmeans[indices], qmeans_basX[i][indices])[0])

                print('\tSpearman correlation for exceptional rankings:')
                for i in range(len(qmeans_basX)):
                    print(f'\t\tspearman_exceptional_bas{i} ' + ' | '.join(map(lambda x: f'{x:.4f}', spearman_exceptional_basX[i])))
                print('\t\t'+'-'*20)
                print('\t\tspearman_exceptional_inv ' + ' | '.join(map(lambda x: f'{x:.4f}', spearman_exceptional_inv)))

                # %% [markdown]
                # ### 3. Save

                # %%
                np.savez(os.path.join(PROJ_DIR, 'results', FILENAME.replace('_measures','_results_' + TARGET_MEASURE)), \
                        correct_pairings_inv=correct_pairings_inv, \
                        correct_pairings_basX=correct_pairings_basX, \
                        spearman_inv=spearman_inv, \
                        spearman_basX=spearman_basX, \
                        aucs_inv=aucs_inv, \
                        aucs_basX=aucs_basX, \
                        spearman_exceptional_inv=spearman_exceptional_inv, \
                        spearman_exceptional_basX=spearman_exceptional_basX, \
                        boundaries=boundaries)
