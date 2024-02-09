import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os.path
import numpy as np
import copy
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import PIL
PROJ_DIR = os.path.realpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PATH_DIR = './'
NUM_SAMPLES = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#https://nextjournal.com/gkoehler/pytorch-mnist
class MNISTClassifier(nn.Module):
    MODEL_LABEL_NUM = 10
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, MNISTClassifier.MODEL_LABEL_NUM)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim = -1)
    
class MLP(torch.nn.Module):
    def __init__(self, num_features, num_labels, n_neurons):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(num_features, n_neurons)
        self.ac1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_neurons, num_labels)
        self.ac2 = torch.nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.ac1(x)
        logits = self.fc2(x)
        x = self.ac2(logits)
        return x

class MLPLarge(torch.nn.Module):
    def __init__(self, num_features, num_labels, n_neurons, return_logits = False):
        assert len(n_neurons) == 4, 'Four hidden layers are needed'
        super(MLPLarge, self).__init__()
        self.fc1 = torch.nn.Linear(num_features, n_neurons[0])
        self.ac1 = torch.nn.ReLU()
        self.d1 = torch.nn.Dropout(0.7)
        self.fc2 = torch.nn.Linear(n_neurons[0], n_neurons[1])
        self.ac2 = torch.nn.ReLU()
        self.d2 = torch.nn.Dropout(0.7)
        self.fc3 = torch.nn.Linear(n_neurons[1], n_neurons[2])
        self.ac3 = torch.nn.ReLU()
        self.d3 = torch.nn.Dropout(0.7)
        self.fc4 = torch.nn.Linear(n_neurons[2], n_neurons[3])
        self.ac4 = torch.nn.ReLU()
        self.d4 = torch.nn.Dropout(0.5)
        self.fc_out = torch.nn.Linear(n_neurons[-1], num_labels)
        self.ac_out = torch.nn.Softmax(dim=-1)
        self.return_logits = return_logits
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.ac1(x)
        x = self.d1(x)
        x = self.fc2(x)
        x = self.ac2(x)
        x = self.d2(x)
        x = self.fc3(x)
        x = self.ac3(x)
        x = self.d3(x)
        x = self.fc4(x)
        x = self.ac4(x)
        x = self.d4(x)
        logits = self.fc_out(x)
        x = logits if self.return_logits else self.ac_out(logits)
        return x
    
class LogitToOHEWrapper(torch.nn.Module):
    def __init__(self, network, weights='DEFAULT', device='cpu'):
        super(LogitToOHEWrapper, self).__init__()
        self.network = network

    def forward(self, x):
        logits = self.network(x)
        # Apply softmax to convert logits to probabilities
        probabilities = F.softmax(logits, dim=1)
        return probabilities
    
class CIFARResnet50Wrapper(torch.nn.Module):
    def __init__(self, output_logits = False, device='cpu'):
        super(CIFARResnet50Wrapper, self).__init__()
        # Load the pre-trained VGG16 model
        self.network = torchvision.models.resnet50().to(device)#torchvision.models.vgg16().to(device)
        self.dense_out = torch.nn.Linear(1000, 100, device = device)
        self.output_logits = output_logits

    def forward(self, x):
        # Forward pass through the pre-trained ResNet50
        x = self.network(x)
        logits = self.dense_out(x)
        # Apply softmax to convert logits to probabilities
        if self.output_logits:
            return logits
        probabilities = F.softmax(logits, dim=1)
        return probabilities

def load_pretrained_imagenet_model(arch = 'resnet50', use_logits = False):
    if arch == 'vgg16':
        print('Loading VGG16')
        network = torchvision.models.vgg16(weights="IMAGENET1K_V1").to(device).eval()
    elif arch == 'resnet50':
        print('Loading Resnet50')
        network = torchvision.models.resnet50(weights="DEFAULT").to(device).eval()
    elif arch == 'resnet18':
        print('Loading Resnet18')
        network = torchvision.models.resnet18(weights="DEFAULT").to(device).eval()
    elif arch == 'maxvit_t':
        print('Loading MaxViT_t')
        network = torchvision.models.maxvit_t(weights="IMAGENET1K_V1").to(device).eval()
    elif arch == 'vit_b_32':
        print('Loading ViT_b_32')
        network = torchvision.models.vit_b_32(weights="IMAGENET1K_V1").to(device).eval()
    else:
        raise Exception('ERROR: Unknown imagenet architecture', arch)
    if use_logits:
        return network
    return LogitToOHEWrapper(network).eval()
        

def load_pretrained_mnist_model(path):
    network = MNISTClassifier()

    if os.path.isfile(path):
        network.load_state_dict(torch.load(path))
        network.eval()
        network.to(device)
    else:
        raise Exception('ERROR: Could not find model at ',path)
    return network

def load_pretrained_cifar_model(path):
    network = CIFARResnet50Wrapper(output_logits=False)

    if os.path.isfile(path):
        network.load_state_dict(torch.load(path))
        network.eval()
        network.to(device)
    else:
        raise Exception('ERROR: Could not find model at ',path)
    return network

def load_pretrained_mlp_model(path, num_features, num_labels, num_neurons):
    network = MLP(num_features, num_labels, num_neurons)

    if os.path.isfile(path):
        network.load_state_dict(torch.load(path))
        network.eval()
        network.to(device)
    else:
        raise Exception('ERROR: Could not find model at ',path)
    return network

def load_pretrained_mlp_large_model(path, num_features, num_labels, num_neurons):
    network = MLPLarge(num_features, num_labels, num_neurons)

    if os.path.isfile(path):
        network.load_state_dict(torch.load(path))
        network.eval()
        network.to(device)
    else:
        raise Exception('ERROR: Could not find model at ',path)
    return network

IMAGENETTE_PATH = os.path.join(PROJ_DIR, 'data', 'imagenette')
IMAGENETTE_CLASS_DICT = {'n01440764':0, 'n02102040':217, 'n02979186':481, 'n03000684':491, 'n03028079':497, 'n03394916':566, 'n03417042':569, 'n03425413':571, 'n03445777':574, 'n03888257':701}
IMAGENETTE_CLASS_DIRS = sorted(list(IMAGENETTE_CLASS_DICT.keys()))

def get_imagenette_dataset(is_test=False, project_path:str='../'):
    ''' Loads the imagenette dataset. By default it loads the train partition, unless otherwise indicated'''
    def transform_labels(l):
        new_l = IMAGENETTE_CLASS_DICT[IMAGENETTE_CLASS_DIRS[l]]
        return new_l

    def load_sample(path: str) -> dict:
        """Read data as image and path. """
        return PIL.Image.open(path).convert("RGB")


    DATA_TRAIN_PATH = os.path.join(project_path, IMAGENETTE_PATH, 'train')
    DATA_TEST_PATH = os.path.join(project_path, IMAGENETTE_PATH, 'val')

    transform = transforms.Compose([
                    transforms.Resize(256), 
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
    # Load test data and make loaders.
    dataset = torchvision.datasets.DatasetFolder(DATA_TEST_PATH if is_test else DATA_TRAIN_PATH, 
                                                loader=load_sample, 
                                                is_valid_file=lambda path: path[-5:]==".JPEG",
                                                transform=transform, # Should we do this here or work with the full images for the RL process??
                                                target_transform=transform_labels)
    return dataset

def get_mnist_dataset(is_test=False, project_path:str='../'):
    mnist_path = os.path.join(project_path, 'data', 'mnist')
    return torchvision.datasets.MNIST(mnist_path, train=not is_test, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ]))

def get_cifar_dataset(is_test=False, project_path:str='../'):
    cifar_path = os.path.join(project_path, 'data', 'cifar')
    return torchvision.datasets.CIFAR100(cifar_path, train=not is_test, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        mean=[0.5071, 0.4867, 0.4408],std=[0.2675, 0.2565, 0.2761])
                                    ]))

def get_image_loader(is_test:bool, dataset_name:str, batch_size:int = 24, project_path:str='../', shuffle:bool = False) -> torch.utils.data.DataLoader:
    if dataset_name == 'imagenet':
        dataset = get_imagenette_dataset(project_path=project_path, is_test=is_test)
    elif dataset_name == 'mnist':
        dataset = get_mnist_dataset(project_path=project_path, is_test=is_test)
    elif dataset_name == 'cifar':
        dataset = get_cifar_dataset(project_path=project_path, is_test=is_test)
    else:
        raise Exception(f'Unknown image dataset: {dataset_name}')
    
    loader = torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
    return loader

def get_image_train_loader(dataset_name:str, batch_size:int = 24, project_path:str='../') -> torch.utils.data.DataLoader:
    return get_image_loader(False, dataset_name, batch_size, project_path)

def get_image_test_loader(dataset_name:str, batch_size:int = 24, project_path:str='../', shuffle:bool = False) -> torch.utils.data.DataLoader:
    return get_image_loader(True, dataset_name, batch_size, project_path, shuffle)

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

def _get_random_ranking_row(dimensions:tuple[int]) -> torch.Tensor:
  '''
  Generates a random ranking row.
  
  A ranking row is like an attribution tensor (it holds a value for each input variable
  that represents its importance) but these values are all different and between 0 and 1 (indicating the position of
  the corresponding variable in the ranking of all variables according to their attribution)
  '''
  num_elems = 1
  for d in dimensions:
    num_elems *= d
  input = np.random.permutation(num_elems).reshape(dimensions)/(num_elems-1)
  return torch.tensor(input, dtype=torch.float32).to(device)

def _get_chunky_random_ranking_row(dimensions:tuple[int], chunk_height:int, chunk_width:int, single_channel:bool) -> torch.Tensor:
    chunky_shape = (1 if single_channel else dimensions[0], dimensions[1]//chunk_height, dimensions[2]//chunk_width)
    random_row = _get_random_ranking_row(chunky_shape)
    upscaled_ranking = F.interpolate(random_row.unsqueeze(0), size=dimensions[1:], mode='nearest').squeeze(0)
    if single_channel:
        upscaled_ranking = upscaled_ranking.repeat(dimensions[0], 1, 1)
    return upscaled_ranking

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
                                       masking_values:torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
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
                              masking_values:torch.Tensor = None,\
                              noisy_inverse:bool = False) -> dict:
    '''
    Given an input, a target output, a model and a ranking (ordering of the input variables, indicated with values between 0 and 1), computes:
      - Output curve with as many points as indicated by num_samples
      - is_hit_curve with as many points as indicated by num_samples. Each element is True only if the target output is the largest of all outputs
      - The measures indicated in the measures parameter:
        - mean - Average activation in the num_samples points
        - auc - AUC of the activation curve in the num_samples points
        - at_first_argmax - Activation of the target output at the first selection level in which the target is the largest output

    A ranking row is like an attribution tensor (it holds a value for each input variable
    that represents its importance) but these values are all different and between 0 and 1 (indicating the position of
    the corresponding variable in the ranking of all variables according to their attribution)
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
        inverse_ranking = 1 - ranking_row
        if noisy_inverse:
            attributions = inverse_ranking + _get_random_ranking_row(ranking_row.shape)
            inverse_ranking = _attributions_to_ranking_row(attributions.flatten().detach().cpu().numpy())
            inverse_ranking = torch.tensor(inverse_ranking.reshape(ranking_row.shape)).to(device)
        result_inverse = get_measures_for_ranking(input, inverse_ranking, output_label, model, measures, num_samples, with_inverse=False, with_random=False, masking_values=masking_values)
        result['output_curve_inv'] = result_inverse['output_curve']
        result['is_hit_curve_inv'] = result_inverse['is_hit_curve']
        result['inverse_ranking'] = inverse_ranking
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


if __name__ == '__main__':
    attributions = np.array([8.0,3.2,0.1,3.2,3.2])
    r1 = _attributions_to_ranking_row(attributions, reverse = False)
    print(r1)
    i2 = np.argsort(attributions)
    v2 = np.linspace(0,1,attributions.size)
    r2 = np.empty_like(v2)
    r2[i2] = v2
    equal = attributions[i2[:-1]]==attributions[i2[1:]]
    print(i2)
    print(equal)
    i_equal = np.where(equal)[0] + 1
    print(i_equal)
    print(attributions[i2[i_equal]])
    r2[i2[i_equal]] = 7
    print(r2)