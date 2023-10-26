import os
import sys
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Import library from src as fl (faithfulness library)
PROJ_DIR = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(PROJ_DIR,'src'))
import xai_faithfulness_experiments_lib_edits as fl

# Tests _attributions_to_ranking_row
test_attributions = np.array([0.55, 0.3, 1.4, -3.2, -4])
correct_ranking = np.array([0.75, 0.5,  1.0, 0.25, 0.0])
np.testing.assert_array_equal(fl._attributions_to_ranking_row(test_attributions), correct_ranking)

# Tests _get_random_ranking_row
for dimensions in [(10,), (28,28,), (28,28,1,)]:
    random_ranking = fl._get_random_ranking_row(dimensions)
    assert random_ranking.shape == dimensions
    assert random_ranking.max() == 1
    assert random_ranking.min() == 0
    assert random_ranking.unique().numel() == random_ranking.numel()
    assert random_ranking.dtype == torch.float32


## TESTS WITH AVILA DATASET
DATASET_PATH = os.path.join(PROJ_DIR,'assets', 'data', f'avila.npz')
# Load dataset
file_data = np.load(DATASET_PATH)
x_train = file_data['x_train']
x_test = file_data['x_test']
y_train = file_data['y_train']
y_test = file_data['y_test']
NUM_CLASSES_AVILA = len(np.unique(y_train))
MODEL_PATH = os.path.join(PROJ_DIR,'assets', 'models', f'avila-mlp.pth')
network = fl.load_pretrained_mlp_model(MODEL_PATH, x_train.shape[1], NUM_CLASSES_AVILA, 100)

masking_values = torch.zeros(x_train.shape[1:], dtype=torch.float32).to(device)
test_ranking = torch.tensor([0.7778, 0.6667, 0.5556, 0.0000, 0.2222, 0.8889, 0.4444, 0.1111, 1.0000, 0.3333], dtype=torch.float32)

x_sample_tensor = torch.tensor(x_train[0], dtype=torch.float32).to(device)
measures = fl.get_measures_for_ranking(x_sample_tensor, test_ranking, torch.tensor(y_train[0]).to(device), network, num_samples=20, with_inverse=True, with_random=False, masking_values=masking_values)
correct_curve = np.array([0.9999999, 0.9999999, 0.9999999, 0.9999951, 0.9999951, 0.9998678 , 0.9998678 , 0.99932206, 0.99932206, 0.99961853, 0.99961853, 0.99660987, 0.99660987, 0.014372  , 0.014372, 0.9999324 , 0.9999324 , 0.9999529 , 0.9999529 , 0.9999435 ], dtype=np.float32)
np.testing.assert_array_almost_equal(measures['output_curve'], correct_curve)
correct_hit_curve = np.array([ True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True, False, False,  True,  True,  True, True,  True])
np.testing.assert_array_equal(measures['is_hit_curve'], correct_hit_curve)
np.testing.assert_almost_equal(measures['mean'], 0.90096414)
np.testing.assert_almost_equal(measures['at_first_argmax'], 0.9999999)
np.testing.assert_almost_equal(measures['auc'], 17.019312)
correct_inv_curve = np.array([0.99999964, 0.99999964, 0.99999964, 0.99999917, 0.99999917, 1.        , 1.        , 1.        , 1.        , 1.        , 1.        , 0.99999845, 0.99999845, 0.99994564, 0.99994564, 0.999964  , 0.999964  , 0.9999912 , 0.9999912 , 0.9999435 ], dtype=np.float32)
np.testing.assert_array_almost_equal(measures['output_curve_inv'], correct_inv_curve)
correct_inv_hit_curve = np.array([ True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True])
np.testing.assert_array_equal(measures['is_hit_curve_inv'], correct_inv_hit_curve)
np.testing.assert_almost_equal(measures['mean_inv'], -0.09902269)
np.testing.assert_almost_equal(measures['at_first_argmax_inv'], 2.3841858e-07)
np.testing.assert_almost_equal(measures['auc_inv'], -1.9804554)

import time

'''start_time = time.time()
for i in range(1000):
    x_sample_tensor = torch.tensor(x_train[i], dtype=torch.float32).to(device)
    measures = fl.get_measures_for_ranking(x_sample_tensor, test_ranking, y_train[i], network, num_samples=20, with_inverse=True, with_random=False, masking_values=masking_values)

print(time.time() - start_time)
'''
## TESTS WITH MNIST DATASET
DATASET_PATH = os.path.join(PROJ_DIR,'assets', 'data', f'avila.npz')
# Load dataset
import torchvision

batch_size = 256

MNIST_PATH = os.path.join(PROJ_DIR, 'data', 'mnist')

train_loader = torch.utils.data.DataLoader(
                                            torchvision.datasets.MNIST(MNIST_PATH, train=True, download=True,
                                                                        transform=torchvision.transforms.Compose([
                                                                        torchvision.transforms.ToTensor(),
                                                                        torchvision.transforms.Normalize(
                                                                            (0.1307,), (0.3081,))
                                                                        ])),
                                            batch_size=batch_size, shuffle=True)

examples = enumerate(train_loader)
batch_idx, (x_train, y_train) = next(examples)
MODEL_PATH = os.path.join(PROJ_DIR,'assets', 'models', f'mnist-softmax-mlp.pth')
network = fl.load_pretrained_mnist_model(MODEL_PATH)
masking_values = torch.zeros(x_train.shape[1:], dtype=torch.float32).to(device)

test_ranking = fl._get_random_ranking_row(x_train.shape[1:])

SAMPLE_NUM = 0
row = x_train[SAMPLE_NUM].clone().detach().to(device)
label = y_train[SAMPLE_NUM].clone().detach().to(device)

start_time = time.time()
for i in range(1000):
    measures = fl.get_measures_for_ranking(row, test_ranking, label, network, num_samples=20, with_inverse=True, with_random=False, masking_values=masking_values)

print(time.time() - start_time)