import numpy as np
import os

PROJ_DIR = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))

keys = ['correct_pairings_inv', 'correct_pairings_basX', 'spearman_inv', 'spearman_basX', 'aucs_inv', 'aucs_basX', 'spearman_exceptional_inv', 'spearman_exceptional_basX']

tuples_to_test = [('imagenet', 'resnet18'), \
                      ('imagenet', 'resnet50'), \
                      ('imagenet', 'vgg16'), \
                      #('cifar', 'resnet50'), \
                      #('mnist', 'softmax'), \
                    ]

#tuples_to_test = [('cmnist', 'resnet18')]

#target_measures = ['AttributionLocalisation', 'TopKIntersection', 'RelevanceRankAccuracy', 'RelevanceMassAccuracy', 'AUC']
#target_measures = ['LocalLipschitzEstimate', 'RelativeInputStability', 'RelativeOutputStability', 'MaxSensitivity', 'AvgSensitivity']
target_measures = ['FaithfulnessCorrelation', 'FaithfulnessEstimate', 'MonotonicityCorrelation', 'Sufficiency']
#target_measures = ['EfficientMPRT']
GENERATION_MODE = '_randomattr'
#SUFFIX = '_localization_s_area'
#SUFFIX = '_Robustness'
SUFFIX = '_quantus_other'
#SUFFIX = '_EfficientMPRT'

# TABLE3
# tuples_to_test = [('20newsgroups-truncated', ''), \
#                       ('mnist', 'softmax'), \
#                       ('cifar', 'resnet50'), \
#                       ('imagenet', 'resnet18'), \
#                       ('imagenet', 'resnet50'), \
#                       ('imagenet', 'vgg16'), \
#                       ('imagenet', 'vit_b_32'), \
#                       ('imagenet', 'maxvit_t')
#                     ]
# target_measures = ['']
# GENERATION_MODE = ''
# SUFFIX = ''

# TABLE4
# tuples_to_test = [('cifar', 'resnet50'), \
#                       ('imagenet', 'resnet18'), \
#                       ('imagenet', 'resnet50'), \
#                       ('imagenet', 'vgg16'), \
#                       ('imagenet', 'vit_b_32'), \
#                       ('imagenet', 'maxvit_t')
#                     ]
# target_measures = ['']
# GENERATION_MODE = '_chunky'
# SUFFIX = ''

# TABLE5
# tuples_to_test = [('avila', '0'), \
#                       ('avila', 'ood-mean'), \
#                       ('avila', 'ood-zeros'), \
#                       ('avila', 'undertrained'), \
#                       ('avila', 'untrained'), \
#                       ('glass', '0'), \
#                       ('glass', 'ood-mean'), \
#                       ('glass', 'ood-zeros'), \
#                       ('glass', 'undertrained'), \
#                       ('glass', 'untrained')
#                     ]
# target_measures = ['']
# GENERATION_MODE = ''
# SUFFIX = ''

# TABLE6
# tuples_to_test = [('imagenet', 'resnet18'), \
#                       ('imagenet', 'vgg16'), \
#                     ]
# target_measures = ['FaithfulnessCorrelation', 'FaithfulnessEstimate', 'MonotonicityCorrelation', 'Sufficiency']
# GENERATION_MODE = '_randomattr'
# SUFFIX = '_quantus_other'

# TABLE7
# tuples_to_test = [('cmnist', 'resnet18')]
# target_measures = ['AttributionLocalisation', 'TopKIntersection', 'RelevanceRankAccuracy', 'RelevanceMassAccuracy', 'AUC']
# GENERATION_MODE = '_randomattr'
# SUFFIX = '_localization_s_area'

print(f'Dataset - Model\tDeltaPA\tDeltaRho')

for TARGET_MEASURE in target_measures:
    print(r'''\hline
        \multicolumn{4}{|c|}{''' + TARGET_MEASURE + r'''}\\
        \hline''')
    for DATASET_NAME, MODEL_NAME in tuples_to_test:
        if 'vit' in MODEL_NAME:
            GENERATION_MODE = '_random'
            if 'maxvit' in MODEL_NAME:
                GENERATION_MODE = '_random_b'
        num_files = 0 # Count how many files are involved for use below
        result_dict = {}
        for f in os.listdir(os.path.join(PROJ_DIR, 'results')):
            if f.startswith(DATASET_NAME) and f.endswith(f"{MODEL_NAME}{GENERATION_MODE}{SUFFIX}_results{'_' + TARGET_MEASURE if TARGET_MEASURE != '' else ''}.npz"):#{MODEL_NAME}{GENERATION_MODE}_localization_s_area_results{TARGET_MEASURE}.npz
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
            result_dict[f'{k}_mean'] = np.nanmean(result_dict[k], axis=0)
            result_dict[f'{k}_std'] = np.nanstd(result_dict[k], axis=0)
        
        delta_p = result_dict['correct_pairings_inv_mean']-result_dict['correct_pairings_basX_mean'][0]
        #print('delta_p', delta_p)

        delta_ps = result_dict['correct_pairings_inv']-result_dict['correct_pairings_basX'][:,0]
        delta_p_mean = delta_ps.mean()
        delta_p_std = delta_ps.std()

        delta_rho = result_dict['spearman_inv_mean']-result_dict['spearman_basX_mean'][0]
        delta_rhos = result_dict['spearman_inv']-result_dict['spearman_basX'][:,0]
        delta_rho_mean = delta_rhos.mean()
        delta_rho_std = delta_rhos.std()
        #print('delta_rho', delta_rho)
        #print(f'{DATASET_NAME} & {MODEL_NAME} & {delta_p[0]:.3f} & {delta_rho[0]:.3f}\\\\')
        print(f'{DATASET_NAME} & {MODEL_NAME} & {delta_p_mean:.3f}$\\pm${delta_p_std:.3f} & {delta_rho_mean:.3f}$\\pm${delta_rho_std:.3f}\\\\')