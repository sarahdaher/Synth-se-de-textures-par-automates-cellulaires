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
    return torch.rand(size, C, H, W)


def sample_pool(pool, batch_size):
    """
At each training step we sample a few states from the pool and replace one of them with an empty state, so the model doesn't forget how to build the pattern from scratch (citee du papier)

    pioche batch_size etats random dans le pool et  force 1 a etre reinitialisé avec du bruit

    on return batch (batch_size, C, H, W), idx (liste d'indices) 
    """
    
    idx      = random.sample(range(len(pool)), batch_size)
    batch    = pool[idx].clone() #"At each training step we sample a few states from the pool and replace one of them with an empty state" , clone c'est le copy de torch
    batch[0] = torch.rand(batch[0].shape)   #jai choisi le premier au pif car aleatoire 
    return batch, idx


def train(nca, target_grams, steps, batch, H, W):
    """
    boucle d'entrainement 

    lr : 2e-3 jusqu'au step 2000, puis 2e-4 (cf papier)

    """

    #regarder texture_loss dans loss.py et mettre les bons parametres

    pool = make_pool()
    optimizer = optim.Adam(nca.parameters(), lr=2e-3)
    for step in range(steps):

        if step == 2000:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 2e-4

        states, idx = sample_pool(pool, batch)
        n=random.randint(32, 64) 
        states = nca(states, steps=n) 
        loss = texture_loss(states[:,:3,:,:], target_grams)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pool[idx] = states #maj du pool avec les nouveaux etats
        if step % 200 == 0:
            print(f"step {step} ; loss {loss.item():.4f}")
            save_image(states[:,:3,:,:], f"{OUT_DIR}/step_{step}.png") #enregistrer que rgb pour visualiser les images
    return nca
