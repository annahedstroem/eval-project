# Import the necessary libraries
import sys
import os
PROJ_DIR = os.path.realpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJ_DIR,'src'))
import xai_faithfulness_experiments_lib_edits as fl

import numpy as np
from tqdm import tqdm
import quantus
from quantus.metrics.robustness import LocalLipschitzEstimate, RelativeInputStability, RelativeOutputStability, MaxSensitivity, AvgSensitivity
from captum.attr import Saliency, IntegratedGradients, InputXGradient, LRP, GuidedBackprop, Deconvolution, LayerAttribution, LayerGradCam, GuidedGradCam

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using {device}')

robustness_functions = {'LocalLipschitzEstimate': LocalLipschitzEstimate(
                                                                            nr_samples=10,
                                                                            perturb_std=0.2,
                                                                            perturb_mean=0.0,
                                                                            norm_numerator=quantus.similarity_func.distance_euclidean,
                                                                            norm_denominator=quantus.similarity_func.distance_euclidean,    
                                                                            perturb_func=quantus.perturb_func.gaussian_noise,
                                                                            similarity_func=quantus.similarity_func.lipschitz_constant,
                                                                            disable_warnings=True
                                                                        ),
                        'RelativeInputStability': RelativeInputStability(
                                nr_samples=5,
                                disable_warnings=True
                            ),
                        'RelativeOutputStability': RelativeOutputStability(
                                nr_samples=5,
                                disable_warnings=True
                            ),
                        'MaxSensitivity': MaxSensitivity(
                                                            nr_samples=10,
                                                            lower_bound=0.2,
                                                            norm_numerator=quantus.norm_func.fro_norm,
                                                            norm_denominator=quantus.norm_func.fro_norm,
                                                            perturb_func=quantus.perturb_func.uniform_noise,
                                                            similarity_func=quantus.similarity_func.difference,
                                                            disable_warnings=True
                                                        ),
                        'AvgSensitivity': AvgSensitivity(
                                                            nr_samples=10,
                                                            lower_bound=0.2,
                                                            norm_numerator=quantus.norm_func.fro_norm,
                                                            norm_denominator=quantus.norm_func.fro_norm,
                                                            perturb_func=quantus.perturb_func.uniform_noise,
                                                            similarity_func=quantus.similarity_func.difference,
                                                            disable_warnings=True
                                                        )
                        }

def inverse_wrapper(model, inputs, targets, **kwargs):
    explain_func = kwargs["explain_func"]
    a_batch = explain_func(model, inputs, targets, **kwargs)
    a_batch_inv = fl.get_inverse_batched(a_batch)
    return a_batch_inv

def compute_measures_for_sample(network:torch.nn.Module,\
                                row:torch.Tensor,\
                                label:torch.Tensor,\
                                num_rankings: int) -> dict:
        # To be used by Quantus
        x_batch_pt = torch.unsqueeze(row, dim=0)
        x_batch = x_batch_pt.to('cpu').numpy()
        y_batch = torch.unsqueeze(label, dim=0).to('cpu').numpy()

        attribution_functions = [lambda model, inputs, targets, **kwargs: Saliency(model).attribute(inputs=torch.tensor(inputs).to(device), target=torch.tensor(targets).to(device)).cpu().numpy(),\
                                lambda model, inputs, targets, **kwargs: IntegratedGradients(model).attribute(inputs=torch.tensor(inputs).to(device), target=torch.tensor(targets).to(device), baselines=torch.zeros_like(x_batch_pt)).cpu().numpy(),\
                                lambda model, inputs, targets, **kwargs: InputXGradient(model).attribute(inputs=torch.tensor(inputs).to(device), target=torch.tensor(targets).to(device)).cpu().numpy(),\
                                lambda model, inputs, targets, **kwargs: GuidedBackprop(model).attribute(inputs=torch.tensor(inputs).to(device), target=torch.tensor(targets).to(device)).cpu().numpy(),\
                                lambda model, inputs, targets, **kwargs: Deconvolution(model).attribute(inputs=torch.tensor(inputs).to(device), target=torch.tensor(targets).to(device)).cpu().numpy()]

        attributions = list(map(lambda f: f(network,
                                            x_batch,
                                            y_batch)[0], attribution_functions))

        all_rankings = np.array(attributions)
        num_rankings = all_rankings.shape[0]

        # Dict to store all results
        all_measures = {}

        all_measures['ranking'] = all_rankings
        print(f'Computing measures...')
        

        for metric in robustness_functions:
            all_measures[metric] = np.zeros(num_rankings, dtype=np.float32)
            all_measures[metric + '_inv'] = np.zeros(num_rankings, dtype=np.float32)

        for i,f in enumerate(attribution_functions):
            #For each ranking, retrieve and store Quantus' faithfulness metrics
            a_batch = np.expand_dims(all_rankings[i], 0)
            a_batch_inv = np.expand_dims(fl.get_inverse(all_rankings[i]), 0)
            #print('x_batch shape:',x_batch.shape)
            #print('y_batch shape:',y_batch.shape)
            #print('a_batch shape:',a_batch.shape)
            #print('network(x_batch) shape:',network(torch.tensor(x_batch).to(device)).shape)
            #print(a_batch)
            for metric in robustness_functions:
                robustness_function = robustness_functions[metric]
                all_measures[metric][i] = robustness_function(model=network,
                                                                    x_batch=x_batch, 
                                                                    y_batch=y_batch,
                                                                    a_batch=a_batch,
                                                                    explain_func=attribution_functions[i],
                                                                    device=device,
                                                                    channel_first=True)[0]
                all_measures[metric + '_inv'][i] = robustness_function(model=network, 
                                                                    x_batch=x_batch, 
                                                                    y_batch=y_batch,
                                                                    a_batch=a_batch_inv,
                                                                    explain_func=inverse_wrapper,
                                                                    explain_func_kwargs={'explain_func': attribution_functions[i]},
                                                                    device=device,
                                                                    channel_first=True)[0]
            return all_measures

if __name__ == '__main__':
    NUM_RANKINGS = 10000
    GENETIC_ITERATIONS = 50
    NEEDED_SAMPLES = 5
    SAMPLE_INDICES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 1, 11, 21, 31, 41, 51, 61, 71, 81, 91]

    # For GENERATORS, empty means random. '_genetic', '_captum', Another alternative is '_full', that generates all possible rankings
    tuples_to_test = [#('avila', '', ['_full']), \
                      #('glass', '', ['_full']), \
                      ('imagenet', 'resnet18'), \
                      ('imagenet', 'resnet50'), \
                      ('imagenet', 'vgg16'), \
                      ('cifar', 'resnet50'), \
                      ('mnist', 'softmax'), \
                    ]

    for DATASET, MODEL_NAME in tuples_to_test:
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
                all_measures = compute_measures_for_sample(network, row, label, NUM_RANKINGS)

            quantus_results = {}
            for metric in robustness_functions:
                quantus_results[metric] = all_measures[metric]
                quantus_results[metric + '_inv'] = all_measures[metric + '_inv']

            np.savez(os.path.join(PROJ_DIR, 'results', f'{DATASET}_{sample_index}_{MODEL_NAME}_Robustness_measures.npz'), \
                    row=row.to('cpu').numpy(), \
                    label=label.to('cpu').numpy(), \
                    rankings=all_measures['ranking'], \
                    **quantus_results)
