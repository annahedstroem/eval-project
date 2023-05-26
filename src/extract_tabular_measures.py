#!/usr/bin/env python
# coding: utf-8

# Import the necessary libraries
import sys
import os
PROJ_DIR = os.path.realpath(os.path.dirname(os.path.abspath('')))
sys.path.append(os.path.join(PROJ_DIR,'src'))
import xai_faithfulness_experiments_lib_edits as fl

import numpy as np

# Avila dataset
DATASET = 'avila'
DATASET_PATH = os.path.join(PROJ_DIR,'assets', 'data', f'{DATASET}.npz')
MODEL_PATH = os.path.join(PROJ_DIR,'assets', 'models', f'{DATASET}-mlp.pth')

# Load dataset
file_data = np.load(DATASET_PATH)
x_train = file_data['x_train']
x_test = file_data['x_test']
y_train = file_data['y_train']
y_test = file_data['y_test']

# Load model
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_NEURONS = 100
MODEL_EPOCHS= 2000
MODEL_LR = 1.0e-1
MODEL_LABEL_NUM = len(np.unique(y_train))

class MLP(torch.nn.Module):
    def __init__(self, n_neurons):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(x_train.shape[1], n_neurons)
        self.ac1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_neurons, MODEL_LABEL_NUM)
        self.ac2 = torch.nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.ac1(x)
        logits = self.fc2(x)
        x = self.ac2(logits)
        return x
    
class MLPForQuantus(torch.nn.Module):
    def __init__(self, n_neurons):
        super(MLPForQuantus, self).__init__()
        self.fc1 = torch.nn.Linear(x_train.shape[1], n_neurons)
        self.ac1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_neurons, MODEL_LABEL_NUM)
        self.ac2 = torch.nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.ac1(x)
        logits = self.fc2(x)
        x = self.ac2(logits)
        #Reshapes to be able to use Quantus
        x = torch.squeeze(x)
        x = torch.unsqueeze(x,dim=0)
        return x

network = MLP(MODEL_NEURONS)
network.load_state_dict(torch.load(MODEL_PATH))
network.eval()
network.to(device)

network_fq = MLPForQuantus(MODEL_NEURONS)
network_fq.load_state_dict(torch.load(MODEL_PATH))
network_fq.eval()
network_fq.to(device)


import itertools

NUM_VARS = x_train.shape[1]
print(NUM_VARS)

permutations = list(itertools.permutations(range(NUM_VARS)))
all_rankings = np.array(permutations) / (NUM_VARS - 1)


from tqdm import tqdm
import quantus

SAMPLE_NUM = 250 # Select one of the training examples in the dataset to be explained

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
        all_measures[k+s] = np.zeros((num_rankings, fl.NUM_SAMPLES), dtype=np.float32 if 'is_hit' in k else bool)
all_measures['ranking'] = np.zeros((num_rankings, NUM_VARS), dtype=np.float32)

# Compute the results for each possible ranking
for i in tqdm(range(num_rankings), miniters=1000):
    #TODO - Add several samples for qbas instead of a single one
    measures = fl.get_measures_for_ranking(row, torch.tensor(all_rankings[i]).to(device), label, network, num_samples=fl.NUM_SAMPLES, with_inverse=True, with_random=True)
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

for i in tqdm(range(num_rankings),  miniters=1000):
    #For each ranking, retrieve and store Quantus' faithfulness metrics
    a_batch = np.expand_dims(np.expand_dims(np.expand_dims(all_rankings[i],0),0),0)
    #print('x_batch shape:',x_batch.shape)
    #print('y_batch shape:',y_batch.shape)
    #print('a_batch shape:',a_batch.shape)
    #print('network(x_batch) shape:',network(torch.tensor(x_batch).to(device)).shape)
    #print(a_batch)
    all_measures['faithfulness_correlation'][i] = quantus.FaithfulnessCorrelation(
                                                    nr_runs=100,  
                                                    subset_size=4,  
                                                    perturb_baseline="black",
                                                    perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                                                    similarity_func=quantus.similarity_func.correlation_pearson,  
                                                    abs=False,  
                                                    return_aggregate=False,
                                                    disable_warnings=True
                                                )(model=network_fq, 
                                                x_batch=x_batch, 
                                                y_batch=y_batch,
                                                a_batch=a_batch,
                                                device=device,
                                                channel_first=True)[0]
    all_measures['monotonicity_correlation'][i] = quantus.MonotonicityCorrelation(
                                                    nr_samples=10,
                                                    features_in_step=1,
                                                    perturb_baseline="uniform",
                                                    perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                                                    similarity_func=quantus.similarity_func.correlation_spearman,
                                                    disable_warnings=True
                                                )(model=network_fq, 
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
                                                )(model=network_fq,
                                                    x_batch=x_batch,
                                                    y_batch=y_batch,
                                                    a_batch=a_batch,
                                                    device=device,
                                                   channel_first=True)[0]



np.savez(os.path.join(PROJ_DIR, 'results', f'{DATASET}_{SAMPLE_NUM}_measures.npz'), \
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
         is_hit_curves_bas=all_measures['is_hit_curve_bas'])

