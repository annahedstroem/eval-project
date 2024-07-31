import numpy as np
import os

PROJ_DIR = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))

keys = ['qmeans']

#### TABLE2
tuples_to_test = [('avila', 'mlp'), \
                      ('avila', 'ood-mean'), \
                      ('avila', 'ood-zeros'), \
                      ('avila', 'undertrained'), \
                      ('avila', 'untrained'), \
                      ('glass', 'mlp'), \
                      ('glass', 'ood-mean'), \
                      ('glass', 'ood-zeros'), \
                      ('glass', 'undertrained'), \
                      ('glass', 'untrained')
                    ]
target_measures = ['qmeans']
GENERATION_MODE = '_full'
SUFFIX = ''
##########

print(f'Dataset - Model\tDeltaPA\tDeltaRho')

for TARGET_MEASURE in target_measures:
    print(r'''\hline
        \multicolumn{4}{|c|}{''' + TARGET_MEASURE + r'''}\\
        \hline''')
    for DATASET_NAME, MODEL_NAME in tuples_to_test:
        num_files = 0 # Count how many files are involved for use below
        result_dict = {}
        for f in os.listdir(os.path.join(PROJ_DIR, 'results')):
            if f.startswith(DATASET_NAME) and f.endswith(f"{MODEL_NAME}{GENERATION_MODE}{SUFFIX}_measures.npz"):#{MODEL_NAME}{GENERATION_MODE}_localization_s_area_results{TARGET_MEASURE}.npz
                FILENAME = os.path.join(PROJ_DIR, 'results', f)
                
                num_files += 1
                with np.load(FILENAME) as data:
                    for k in keys:
                        d = np.expand_dims(data[k], axis=0)
                        if k in result_dict:
                            result_dict[k] = np.vstack((result_dict[k], d))
                        else:
                            result_dict[k] = d
        assert num_files>0, f'No files for {DATASET_NAME} {MODEL_NAME}{GENERATION_MODE} {TARGET_MEASURE}'
        
        for k in keys:
            result_dict[f'{k}_mean'] = np.nanmean(result_dict[k], axis=1)
            result_dict[f'{k}_std'] = np.nanstd(result_dict[k], axis=1)
        
        print(f'{DATASET_NAME} & {MODEL_NAME} & {result_dict["qmeans_std"].mean():.3f}$\\pm${result_dict["qmeans_std"].std():.3f}\\\\')