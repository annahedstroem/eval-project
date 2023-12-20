# Import the necessary libraries
import sys
import os
PROJ_DIR = os.path.realpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJ_DIR,'src'))
import xai_faithfulness_experiments_lib_edits as fl
import numpy as np
import captum_generator as cg
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using {device}')
import pickle

ACTIVATION_THRESHOLD = 0.9
Z_SCORE_THRESHOLD = 4
DESIRED_EXPLANATIONS = 1000
BATCH_SIZE = 256

for DATASET in ['imagenet']:
    for MODEL_NAME in ['resnet18-logits','resnet50-logits', 'vgg16-logits']:
        # Load dataset
        if DATASET == '20newsgroups-truncated':
            DATASET_PATH = os.path.join(PROJ_DIR,'assets', 'data', f'{DATASET}.npz')
            # Load dataset
            file_data = np.load(DATASET_PATH)
            x_train = torch.from_numpy(file_data['x_train']).float().to(device)
            y_train = torch.from_numpy(file_data['y_train']).to(device)
            test_loader = [(x_train, y_train)]
        else:
            #torch.manual_seed(0)
            test_loader = fl.get_image_test_loader(DATASET, BATCH_SIZE, PROJ_DIR, shuffle = True)


        # Load model
        if DATASET.startswith('imagenet'):
            network = fl.load_pretrained_imagenet_model(arch = MODEL_NAME.replace('-logits',''), use_logits = 'logits' in MODEL_NAME)
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

        FILENAME = f'{DATASET}_{MODEL_NAME}_noise_exceptionals.pkl'

        if os.path.isfile(os.path.join(PROJ_DIR, 'results', FILENAME)):
            with open(os.path.join(PROJ_DIR, 'results', FILENAME), 'rb') as fIn:
                results = pickle.load(fIn)
        else:
            # The mean is zero because this dataset is standardized
            num_vars = None
            masking_values = None

            with torch.no_grad():
                valid_elements = []

                for batch_idx, (x_train, y_train) in enumerate(test_loader):
                    print(f'Loaded batch  {batch_idx}')
                    if masking_values is None:
                        masking_values = torch.from_numpy(np.zeros(x_train.shape[1:])).float().to(device)
                        num_vars  = 1
                        for d in x_train.shape[1:]:
                            num_vars *= d
                        num_samples = min(fl.NUM_SAMPLES, num_vars)
                        input_shape = x_train.shape[1:]
                    # Find elements from the batch that activate the network enough
                    outputs = network(x_train.to(device))

                    if 'logits' in MODEL_NAME:
                        outputs = torch.softmax(outputs, dim = -1)

                    activated_indices = (outputs[torch.arange(x_train.shape[0]), y_train]>ACTIVATION_THRESHOLD).nonzero().flatten()

                    for i, sample_index in enumerate(activated_indices):
                        if (i+1) % 10 == 0:
                            print(f' Exploring activating sample {i+1}/{activated_indices.size()[0]}')
                        row = x_train[sample_index.item()].to(device)
                        label = y_train[sample_index.item()].to(device)
                    
                        #Compute 100 random rankings to compute the average q
                        qmeans = []
                        for _ in range(100):
                            measures = fl.get_measures_for_ranking(row, fl._get_random_ranking_row(row.shape), label, network, with_inverse=False, with_random=False, masking_values=masking_values)
                            qmeans.append(measures['mean'])
                        qmean_mean = np.mean(qmeans)
                        qmean_std = np.std(qmeans)
                
                        #Grab captum generated explanations and check their z-index
                        captum_rankings = torch.tensor(cg.generate_rankings(row, label, network)).to(device)
                        for method_index, ranking in enumerate(captum_rankings):
                            measures = fl.get_measures_for_ranking(row, ranking, label, network, with_inverse=False, with_random=False, masking_values=masking_values)
                            zscore = (measures['mean'] -  qmean_mean) /  qmean_std
                            if zscore > 4:
                                valid_elements.append({'row': row,\
                                                    'ranking': ranking,\
                                                    'label': label,\
                                                    'qmean_mean': qmean_mean,\
                                                    'qmean_std': qmean_std,\
                                                    'method': method_index
                                                    })
                                print(f'{len(valid_elements)}/{DESIRED_EXPLANATIONS}')
                                if len(valid_elements) >= DESIRED_EXPLANATIONS:
                                    break
                        if len(valid_elements) >= DESIRED_EXPLANATIONS:
                            break

            results = []
            for v in valid_elements:
                row  = v['row']
                ranking = v['ranking']
                label = v['label']
                measures = fl.get_measures_for_ranking(row, ranking, label, network, with_inverse=True, with_random=True, masking_values=masking_values, noisy_inverse=True)
                v['qmean'] = measures['mean']
                v['qinv'] = measures['mean_inv']
                v['qbas'] = measures['mean_bas']
                v['inverse_ranking'] = measures['inverse_ranking']
                results.append(v)
            with open(os.path.join(PROJ_DIR, 'results', FILENAME), 'wb') as fOut:
                pickle.dump(results, fOut)
