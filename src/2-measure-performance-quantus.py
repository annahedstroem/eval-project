# %% [markdown]
# # Measure performance
# This notebook loads a file with precomputed measures for metrics available in Quantus (*faithfulness_correlations* & *monotonicity_correlations*) for a set of rankings for a given instance of the dataset and measures the performance of the different alternative measures
# 
# ## 1. Load libraries, model and data

# Import the necessary libraries
import sys
import os
PROJ_DIR = os.path.realpath(os.path.dirname(os.path.abspath('')))
sys.path.append(os.path.join(PROJ_DIR,'src'))
import xai_faithfulness_experiments_lib_edits as fl
import numpy as np

for FILENAME in os.listdir(os.path.join(PROJ_DIR,'results')):
    if FILENAME.endswith('_measures.npz'):
        print(FILENAME)
        # Load data
        data = fl.load_generated_data(os.path.join(PROJ_DIR, 'results', FILENAME))

        if 'inv_lookup' in data.keys():
            inv_lookup = data['inv_lookup']
        else:
            inv_lookup = np.load(os.path.join(PROJ_DIR, 'results', 'avila_permutations_inv_lookup.npz'))['inv_lookup']

        faithfulness_correlations = data['faithfulness_correlations']
        faithfulness_correlations_basX = []
        monotonicity_correlations = data['monotonicity_correlations']
        monotonicity_correlations_basX = []

        # Compute the baseline with varying number of samples
        def compute_qbas(measure, num_samples):
            random_indices = np.random.randint(0,  measure.shape[0], (measure.shape[0], num_samples))
            random_qmeans = measure[random_indices]
            mean = np.mean(random_qmeans, axis=1)

            # First way to deal with std==0; add some epsilon
            #std = np.std(random_qmeans, axis=1) + 1e-10

            # Second way to deal with std==0; ignore std (divide by 1)
            std = np.std(random_qmeans, axis=1)
            std[std==0] = 1

            # Always ignore std
            std=1
            return (measure - mean) / std

        for i in range(1,11):
            faithfulness_correlations_basX.append(compute_qbas(faithfulness_correlations, i))
            monotonicity_correlations_basX.append(compute_qbas(monotonicity_correlations, i))

        # Compute the qinv version
        def compute_qinv(measure, inv_lookup):
            return measure - measure[inv_lookup]

        faithfulness_correlations_inv = compute_qinv(faithfulness_correlations, inv_lookup)
        monotonicity_correlations_inv = compute_qinv(monotonicity_correlations, inv_lookup)

        MEASURES = [('faithfulness_correlations', faithfulness_correlations, faithfulness_correlations_basX, faithfulness_correlations_inv), ('monotonicity_correlations', monotonicity_correlations, monotonicity_correlations_basX, monotonicity_correlations_inv)]

        level_indices_by_measure = {}
        for name, q, qbasX, qinv in MEASURES:
            # Compute z-score for stratification
            q_mean = np.mean(q)
            q_std = np.std(q)
            z_scores = ((q - q_mean) / q_std).flatten()

            # Stratify z-scores to be able to compare performance on different parts of the spectrum
            indices = np.arange(z_scores.shape[0])
            z_scores_numbered = np.vstack((z_scores, indices))
            level_indices = []
            boundaries = [float('-inf'), 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
            for i in range(1,len(boundaries)+1):
                bottom_limit = boundaries[i-1]
                top_limit = float('inf')
                if i < len(boundaries):
                    top_limit = boundaries[i]
                level_indices.append((z_scores_numbered[:,np.logical_and(bottom_limit<=z_scores, z_scores<top_limit)][1,:].astype(int),(bottom_limit, top_limit)))
            
            level_indices_by_measure[name] = level_indices

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

        correct_pairings_basX_by_measure = {}
        correct_pairings_inv_by_measure = {}
        for name, q, qbasX, qinv in MEASURES:
            correct_pairings_basX = []
            for i in range(len(qbasX)):
                correct_pairings_basX.append(measure_correct_orderings(q, qbasX[i]))
                print(f'{name}_bas{i+1}: {correct_pairings_basX[i]:.4f}')
            correct_pairings_inv = measure_correct_orderings(q, qinv)
            print(f'{name}_inv: {correct_pairings_inv:.4f}')
            
            correct_pairings_basX_by_measure[name] = correct_pairings_basX
            correct_pairings_inv_by_measure[name] = correct_pairings_inv

        # %% [markdown]
        # ### 2.2. Spearman correlation
        # Same thing, is the order of qmeans preserved in qbasX/qinv?

        # %%
        from scipy.stats import spearmanr

        spearman_basX_by_measure = {}
        spearman_inv_by_measure = {}
        for name, q, qbasX, qinv in MEASURES:
            spearman_basX = []
            for i in range(len(qbasX)):
                spearman_basX.append(spearmanr(q, qbasX[i])[0])
                print(f'{name}_bas{i+1}: {spearman_basX[i]:.4f}')
            spearman_inv = spearmanr(q, qinv)[0]
            print(f'{name}_inv: {spearman_inv:.4f}')

            spearman_basX_by_measure[name] = spearman_basX
            spearman_inv_by_measure[name] = spearman_inv

        # %% [markdown]
        # ### 2.3. Ability to detect exceptionally good rankings
        # As stated above, there are some ordering errors in the estimators. Are they in the relevant part of the distribution? i.e. Do they affect the ability to identify exceptionally good rankings?

        # %%
        from sklearn import metrics

        def measure_detection(target_indices, estimator):
            if len(target_indices)==0:
                return 1
            target = np.zeros_like(estimator, dtype=int)
            target[target_indices] = 1
            return metrics.roc_auc_score(target, estimator)

        aucs_basX_by_measure = {}
        aucs_inv_by_measure = {}
        for name, q, qbasX, qinv in MEASURES:
            aucs_inv = []
            aucs_basX = [[] for i in qbasX]

            for indices, (bottom_limit, upper_limit) in level_indices_by_measure[name][2:]:
                aucs_inv.append(measure_detection(indices, qinv))
                for i in range(len(qbasX)):
                    aucs_basX[i].append(measure_detection(indices, qbasX[i]))

            for i in range(len(qbasX)):
                print(f'{name}_auc_bas{i} ' + ' | '.join(map(lambda x: f'{x:.4f}',aucs_basX[i])))
            print(f'{name}_aucs_inv ' + ' | '.join(map(lambda x: f'{x:.4f}',aucs_inv)))

        aucs_basX_by_measure[name] = aucs_basX
        aucs_inv_by_measure[name] = aucs_inv


        # %% [markdown]
        # ### 2.4 Ability to rank exceptionally good rankings
        # How well is the order preserved for exceptionally good rankings?

        # %%
        spearman_exceptional_basX_by_measure = {}
        spearman_exceptional_inv_by_measure = {}
        for name, q, qbasX, qinv in MEASURES:

            spearman_exceptional_inv = []
            spearman_exceptional_basX = [[] for i in qbasX]

            for indices, (bottom_limit, upper_limit) in level_indices_by_measure[name][2:]:
                spearman_exceptional_inv.append(spearmanr(q[indices], qinv[indices])[0])
                for i in range(len(qbasX)):
                    spearman_exceptional_basX[i].append(spearmanr(q[indices], qbasX[i][indices])[0])

            for i in range(len(qbasX)):
                print(f'{name}_spearman_exceptional_bas{i} ' + ' | '.join(map(lambda x: f'{x:.4f}', spearman_exceptional_basX[i])))
            print(f'{name}_spearman_exceptional_inv ' + ' | '.join(map(lambda x: f'{x:.4f}', spearman_exceptional_inv)))

            spearman_exceptional_basX_by_measure[name] = spearman_exceptional_basX
            spearman_exceptional_inv_by_measure[name] = spearman_exceptional_inv

        # %% [markdown]
        # ### 3. Save

        # %%
        for name, q, qbasX, qinv in MEASURES:
            np.savez(os.path.join(PROJ_DIR, 'results', FILENAME.replace('_measures',f'_results_{name}')), \
                    correct_pairings_inv=correct_pairings_inv_by_measure[name], \
                    correct_pairings_basX=correct_pairings_basX_by_measure[name], \
                    spearman_inv=spearman_inv_by_measure[name], \
                    spearman_basX=spearman_basX_by_measure[name], \
                    aucs_inv=aucs_inv_by_measure[name], \
                    aucs_basX=aucs_basX_by_measure[name], \
                    spearman_exceptional_inv=spearman_exceptional_inv_by_measure[name], \
                    spearman_exceptional_basX=spearman_exceptional_basX_by_measure[name])


