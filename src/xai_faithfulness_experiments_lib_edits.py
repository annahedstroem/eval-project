import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os.path
import numpy as np
import copy

PATH_DIR = './'
PATH_PRETRAINED = PATH_DIR + 'mnist-classifier.pth'
NUM_SAMPLES = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_pretrained_model(path = PATH_PRETRAINED):
    #https://nextjournal.com/gkoehler/pytorch-mnist
    class MNISTClassifier(nn.Module):
        def __init__(self):
            super(MNISTClassifier, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x)

    network = MNISTClassifier()

    if os.path.isfile(path):
        network.load_state_dict(torch.load(path))
        network.eval()
        network.to(device)
    else:
        raise Exception('ERROR: Could not find model at ',path)
    return network

'''
    Loads a file that contains a set of feature rankings for a given input
    Returns a dictionary with:
      - input: The input that the rankings try to explain for the one pretrained model that we use
      - label: Label that the rankings try to explain
      - rankings: The actual feature rankings
      - qmeans: One qmean value for each ranking
      - qmean_invs: One qmean value for each inverse ranking
      - qargmaxs: One qmean value for each ranking
      - qargmax_invs: One qmean value for each inverse ranking
      - qaucs: One qauc for each ranking
      - qauc_invs: One qauc for each inverse ranking
      - output_curves: One curve for each ranking representing the output of the label output of the model at the given selection levels
      - is_hit_curves: One curve for each ranking representing whether the output of the label output of the model at the given selection levels is the maximum output of the model
      - output_curves_inv: One curve for each inverse ranking representing the output of the label output of the model at the given selection levels
      - is_hit_curves_inv: One curve for each ranking representing whether the output of the label output of the model at the given selection levels is the maximum output of the model
'''
def load_generated_data_old_format(path):
    data = np.load(path)

    input = data['arr_0']
    label = data['arr_1']
    rankings = data['arr_2']
    plots = data['arr_3']
    inverse_plots = data['arr_4']
    hit_plots = data['arr_5']
    inverse_hit_plots = data['arr_6']
    measures = data['arr_7']
    measures_with_inverse = data['arr_8']

    return {'input': input, \
            'label': label, \
            'rankings': rankings, \
            'qmeans': measures, \
            'qmean_invs': measures_with_inverse, \
            'qargmaxs': None, \
            'qargmax_invs': None, \
            'qaucs': None, \
            'qauc_invs': None, \
            'output_curves': plots, \
            'is_hit_curves': hit_plots, \
            'output_curves_inv': inverse_plots, \
            'is_hit_curves_inv': inverse_hit_plots \
            }

def load_generated_data(path):
    return np.load(path)

''' Q measures:
   - Mean activation
   - Activation at the first point where the label is the argmax of the outputs
   - Activation at a fixed selection point
   - AUC
   '''
def measure_mean_activation(curve):
  return np.mean(curve)

def measure_at_selection_level(curve, selection_level): # Selection level should be in [0, 1]
  selection_point = int(selection_level * curve.shape[0]) # May need to subtract 1 or floor
  return curve[selection_point]

def measure_output_at_first_argmax(curve, is_hit): # Returns output at the first selection point that makes is_hit True.
  selection_point = np.argmax(is_hit) # Finds the first True (returns 0 if there are no Trues)
  if selection_point==0 and not is_hit[0]: # Check if it's zero because there are no Trues
    selection_point=is_hit.size-1
  return curve[selection_point]

def measure_auc(values: np.array, dx: int = 1):
    return np.trapz(values, dx=dx)

''' Utility functions '''
def _get_masked_inputs(original_input, alternative_input, ranking_row, selection_levels):
  '''
  Generates as many masked inputs as selection levels are provided
  Inputs are torch tensors already on device
  '''
  # Reshape selection_levels to be able to broadcast the selection levels and get
  # as many masks as selection levels are provided
  while len(selection_levels.shape) <= len(ranking_row.shape):
    selection_levels = selection_levels.unsqueeze(dim=-1)
  # Compute all masks in batch
  masks = torch.le(ranking_row,selection_levels)
  # Compute masked inputs from masks and original and alternative inputs
  inputs_masked = (original_input*masks) + (alternative_input*torch.logical_not(masks))
  return inputs_masked

def _get_random_ranking_row(dimensions):
  num_elems = 1
  for d in dimensions:
    num_elems *= d
  input = np.random.permutation(num_elems).reshape(dimensions)/(num_elems-1)
  return torch.from_numpy(input)

def _get_class_logits_for_masked_inputs(original_input, alternative_input, ranking_row, selection_levels, model, class_num):
  with torch.no_grad():
    # Send everything to device and work there
    input = original_input.to(device)
    alternative = alternative_input.to(device)
    ranking = ranking_row.to(device)
    levels = selection_levels.to(device)
    inputs = _get_masked_inputs(input, alternative, ranking, levels)
    logits = model(inputs).to('cpu').numpy()
  return logits[:,class_num],np.equal(np.argmax(logits, axis=1),class_num.item())

'''def save_explanation_exploratory_plot(input, curve, is_hit, output_label, filename='unnamed'):
  # Plot and save the figures
  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
  axes[0].imshow(input[0], cmap='gray', interpolation='none')
  axes[0].title.set_text(output_label)
  axes[0].axis("off")
  for i in range(1, len(curve)):
    axes[1].plot([i-1,i],curve[i-1:i+1], lw=5 if is_hit[i] else 1, color='b')
  fig.savefig(f'{filename}.png')
  plt.show()'''

def _get_explanation_exploratory_curve(input:torch.Tensor, \
                                       ranking_row:torch.Tensor, \
                                       num_samples:int, \
                                       output_label:int, \
                                       model:torch.nn.Module, \
                                       masking_values:torch.Tensor = None) -> tuple(torch.Tensor, torch.Tensor):
  '''
  Given an input, a target output, a model and a ranking (ordering of the input variables, indicated with values between 0 and 1)
  computes the activation curve (and is_hit curve) for the target output of the model at as many selection levels as indicated by num_samples

  Selection levels are computed by dividing the total number of attributes in num_sample equally-sized slices in increasing order of attribution/ranking

  Output at a given selection level is computed by performing inference with the model on the input with the input variables up to the selection level
  masked with the given masking_values (should have the same shape as the input; if None is provided, masking_values will be all zeros)

  Returns:
   - Activation curve -> torch.Tensor with shape (num_samples)
   - is_hit curve -> torch.Tensor with shape (num_samples)
  '''
  assert(torch.max(ranking_row)==1.0)
  assert(torch.min(ranking_row)==0.0)
  if masking_values is None:
    masking_values = torch.from_numpy(np.full(input.shape,  0, dtype=np.float32)) #ZEROED-OUT

  # Selection levels
  selection_levels = torch.from_numpy(np.linspace(0, 1, num_samples))

  # Increasing order
  class_logit,is_hit = _get_class_logits_for_masked_inputs(input, masking_values, ranking_row, selection_levels, model, output_label)

  # Compute the numerical value for the measure
  #measure = measure_curves(class_logit)

  return class_logit,is_hit

def _attributions_to_ranking_row(attributions:np.ndarray, \
                                 reverse:bool = False) -> np.ndarray:
    '''
    Returns a unidimensional numpy array r. r_i indicates the order of the i-th variable of the array in the importance ranking
    Ranking position is scaled to [0-1], so 0 is the first element in the ranking and 1 is the last one.

    Arguments:
      - attributions: Unidimensional numpy array that indicates the attribution value for each variable
      - reverse: Indicates whether to order variables in ascending order of attribution value (default) or in descending order.

    Example:
      _attributions_to_ranking_row
    '''
    assert(len(attributions.shape)==1) # This function assumes that the input array is unidimensional
    ranked_attributions = copy.copy(attributions)
    #print(ranked_attributions)
    ranked_attributions = list(enumerate(ranked_attributions))
    ranked_attributions.sort(reverse=reverse, key=lambda x:x[1])
    ranked_attributions = np.array(ranked_attributions)
    #print(ranked_attributions.shape)
    ranking_row = np.zeros(attributions.shape)
    #print(ranked_attributions)
    num_attributes = len(ranked_attributions)
    for i in range(num_attributes):
        x = int(ranked_attributions[i][0])
        ranking_row[x] = i/(num_attributes-1)
    return ranking_row

def get_measures_for_ranking(input:torch.Tensor, \
                              ranking_row:torch.Tensor, \
                              output_label:torch.Tensor, \
                              model:torch.nn.Module, \
                              measures:list[str] = ['mean','at_first_argmax','auc'], \
                              num_samples:int = NUM_SAMPLES, \
                              with_inverse:bool = False, \
                              with_random:bool = False, \
                              masking_values:torch.Tensor = None) -> dict:
    '''
    Given an input, a target output, a model and a ranking (ordering of the input variables, indicated with values between 0 and 1), computes:
      - Output curve with as many points as indicated by num_samples
      - is_hit_curve with as many points as indicated by num_samples. Each element is True only if the target output is the largest of all outputs
      - The measures indicated in the measures parameter:
        - mean - Average activation in the num_samples points
        - auc - AUC of the activation curve in the num_samples points
        - at_first_argmax - Activation of the target output at the first selection level in which the target is the largest output
    '''
    curve,is_hit = _get_explanation_exploratory_curve(input, ranking_row, num_samples, output_label, model, masking_values=masking_values)

    result = {'output_curve': curve, \
              'is_hit_curve': is_hit}

    for measure in measures:
        if measure=='mean':
            result['mean'] = measure_mean_activation(curve)
        elif measure=='at_first_argmax':
            result['at_first_argmax'] = measure_output_at_first_argmax(curve, is_hit)
        elif measure=='auc':
            result['auc'] = measure_auc(curve)

    if with_inverse:
        # Get the measures for the inverse ranking
        result_inverse = get_measures_for_ranking(input, 1-ranking_row, output_label, model, measures, num_samples, with_inverse=False, with_random=False, masking_values=masking_values)
        result['output_curve_inv'] = result_inverse['output_curve']
        result['is_hit_curve_inv'] = result_inverse['is_hit_curve']
        for measure in measures:
            if measure=='mean':
                result['mean_inv'] = result['mean'] - result_inverse['mean']
            elif measure=='at_first_argmax':
                # The selection point is determined by the regular curve
                selection_point = np.argmax(result['is_hit_curve']) # Finds the first True (returns 0 if there are no Trues)
                if selection_point==0 and not is_hit[0]: # Check if it's zero because there are no Trues
                  selection_point=is_hit.size-1
                result['at_first_argmax_inv'] = result['at_first_argmax'] - result_inverse['output_curve'][selection_point]
            elif measure=='auc':
                result['auc_inv'] = result['auc'] - result_inverse['auc']

    if with_random:
        # Get the measures for the inverse ranking
        result_random = get_measures_for_ranking(input, _get_random_ranking_row(ranking_row.shape), output_label, model, measures, num_samples, with_inverse=False, with_random=False, masking_values=masking_values)
        result['output_curve_bas'] = result_random['output_curve']
        result['is_hit_curve_bas'] = result_random['is_hit_curve']
        for measure in measures:
            if measure=='mean':
                result['mean_bas'] = result['mean'] - result_random['mean']
            elif measure=='at_first_argmax':
                # The selection point is determined by the regular curve
                selection_point = np.argmax(result['is_hit_curve']) # Finds the first True (returns 0 if there are no Trues)
                if selection_point==0 and not is_hit[0]: # Check if it's zero because there are no Trues
                  selection_point=is_hit.size-1
                result['at_first_argmax_bas'] = result['at_first_argmax'] - result_random['output_curve'][selection_point]
            elif measure=='auc':
                result['auc_bas'] = result['auc'] - result_random['auc']

    return result

def get_measures_for_attributions(input:torch.Tensor, \
                                  attributions:np.ndarray, \
                                  output_label:torch.Tensor, \
                                  model:torch.nn.Module, \
                                  measures:list[str] = ['mean','at_first_argmax','auc'], \
                                  num_samples:int = NUM_SAMPLES, \
                                  with_inverse:bool = False, \
                                  with_random:bool = False, \
                                  masking_values:torch.Tensor = None) -> dict:
    ''' Same as above, but instead of a ranking, the second parameter contains an attribution value for
    each variable.
    '''
    ranking_row = torch.from_numpy(_attributions_to_ranking_row(attributions))
    return get_measures_for_ranking(input, ranking_row, output_label, model, measures, num_samples, with_inverse, with_random, masking_values=masking_values)
