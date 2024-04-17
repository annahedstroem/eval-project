import numpy as np
import os

PROJ_DIR = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))

keys = ['qmeans']

tuples_to_test = [('imagenet', 'resnet18'), \
                      #('imagenet', 'resnet50'), \
                      #('imagenet', 'vgg16'), \
                      #('cifar', 'resnet50'), \
                      #('mnist', 'softmax'), \
                    ]

#tuples_to_test = [('cmnist', 'resnet18')]

#target_measures = ['AttributionLocalisation', 'TopKIntersection', 'RelevanceRankAccuracy', 'RelevanceMassAccuracy', 'AUC']
#target_measures = ['LocalLipschitzEstimate', 'RelativeInputStability', 'RelativeOutputStability', 'MaxSensitivity', 'AvgSensitivity']
#target_measures = ['FaithfulnessCorrelation', 'FaithfulnessEstimate', 'MonotonicityCorrelation', 'Sufficiency']
target_measures = ['EfficientMPRT']
GENERATION_MODE = ''
#SUFFIX = '_localization_s_area'
#SUFFIX = '_Robustness'
#SUFFIX = '_quantus_other'
SUFFIX = '_EfficientMPRT'

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
tuples_to_test = [('avila', '0'), \
                      ('avila', 'ood-mean'), \
                      ('avila', 'ood-zeros'), \
                      ('avila', 'undertrained'), \
                      ('avila', 'untrained'), \
                      ('glass', '0'), \
                      ('glass', 'ood-mean'), \
                      ('glass', 'ood-zeros'), \
                      ('glass', 'undertrained'), \
                      ('glass', 'untrained')
                    ]
target_measures = ['']
GENERATION_MODE = ''
SUFFIX = ''

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
            if f.startswith(DATASET_NAME) and f.endswith(f'{MODEL_NAME}{GENERATION_MODE}{SUFFIX}_measures{TARGET_MEASURE}.npz'):#{MODEL_NAME}{GENERATION_MODE}_localization_s_area_results{TARGET_MEASURE}.npz
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