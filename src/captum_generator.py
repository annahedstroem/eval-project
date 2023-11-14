import torch
import numpy as np
import os
import sys
PROJ_DIR = os.path.realpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJ_DIR,'src'))
import xai_faithfulness_experiments_lib_edits as fl
from captum.attr import Saliency, IntegratedGradients, InputXGradient, LRP, GuidedBackprop, Deconvolution


def _to_ranking(attributions:np.ndarray) -> np.ndarray:
    original_shape = attributions.shape
    attributions = attributions.flatten()
    ranking = fl._attributions_to_ranking_row(attributions, reverse=True)
    ranking = ranking.reshape(original_shape)
    return ranking

def generate_rankings(row:torch.Tensor, label:torch.Tensor, model:torch.nn.Module) -> np.ndarray:
    '''
    Retrieves from Quantus a few rankings with a few methods
    '''
    x_batch = row.unsqueeze(dim=0)
    y_batch = label.unsqueeze(dim=0)

    attributions = [Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).detach().cpu().numpy()[0],\
                    IntegratedGradients(model).attribute(inputs=x_batch, target=y_batch, baselines=torch.zeros_like(x_batch)).detach().cpu().numpy()[0],\
                    InputXGradient(model).attribute(inputs=x_batch, target=y_batch).detach().cpu().numpy()[0],\
                    GuidedBackprop(model).attribute(inputs=x_batch, target=y_batch).detach().cpu().numpy()[0],\
                    Deconvolution(model).attribute(inputs=x_batch, target=y_batch).detach().cpu().numpy()[0]]##,\
                    #LRP(model).attribute(inputs=x_batch, target=y_batch).detach().cpu().numpy()[0]]
    
    attributions = list(map(_to_ranking, attributions))

    return np.stack(attributions)

if __name__ == '__main__':
    #TESTS
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')
    train_loader = fl.get_imagenette_train_loader(52, PROJ_DIR)

    examples = enumerate(train_loader)
    batch_idx, (x_train, y_train) = next(examples)

    SAMPLE_NUM = 10
    row = x_train[SAMPLE_NUM].clone().detach().to(device)
    label = y_train[SAMPLE_NUM].clone().detach().to(device)

    network = fl.load_pretrained_imagenet_model()

    rankings = generate_rankings(row, label, network)
    print(row.shape)
    print(rankings.shape)
