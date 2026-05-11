import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from config import *
from loss import texture_loss
from utils import save_image


#surtout la section 3.2

def make_pool(size=1024, C=12, H=128, W=128):
    """
    Cree un pool de size etats init avec du bruit U[0,1] ->renvoie tenseur (size, C, H, W)
    """
    # TODO
    #return torch.rand...
    raise NotImplementedError


def sample_pool(pool, batch_size):
    """
At each training step we sample a few states from the pool and replace one of them with an empty state, so the model doesn't forget how to build the pattern from scratch (citee du papier)

    pioche batch_size etats random dans le pool et les force a etre reinitialisé avec du bruit

    on return batch (batch_size, C, H, W), idx (liste d'indices) 
    """
    
    # TODO
    #return batch, idx
    raise NotImplementedError


def train(nca, target_grams, vgg,  steps, batch, H, W, out_dir):
    """
    boucle d'entrainement 

    lr : 2e-3 jusqu'au step 2000, puis 2e-4 (cf papier)

    """
    # TODO
    #regarder texture_loss dans loss.py et mettre les bons parametres


    for step in range(steps):


    
        if step % 200 == 0:
            print(f"step {step} ; loss {loss.item():.4f}")
            save_image(states[:,:3,:,:], f"{out_dir}/step_{step}.png") #enregistrer que rgb pour visualiser les images
    raise NotImplementedError
