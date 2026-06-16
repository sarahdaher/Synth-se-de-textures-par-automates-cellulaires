import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from config import *
from loss import texture_loss, texture_loss_sot
from utils import save_image
from tqdm import tqdm

#surtout la section 3.2

if not MULTI_TEX:

    def make_pool(size=1024, C=12, H=128, W=128):
        """
        Cree un pool de size etats init avec du bruit U[0,1] ->renvoie tenseur (size, C, H, W)
        """
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
    
    
    def train(nca, target, target_grams, steps, batch, H, W, device):
        """
        boucle d'entrainement 

        lr : 2e-3 jusqu'au step 2000, puis 2e-4 (cf papier)

        """

        #regarder texture_loss dans loss.py et mettre les bons parametres

        pool = make_pool().to(device)
        optimizer = optim.Adam(nca.parameters(), lr=2e-3)
        loss_history = []

        for step in tqdm(range(steps)):

            if step == 2000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 2e-4

            states, idx = sample_pool(pool, batch)
            n=random.randint(32, 64) 
            states = nca(states, steps=n)

            rgb = states[:, :3, :, :]

            # switch loss selon config
            if LOSS == "sot":
                loss = texture_loss_sot(rgb, target)
            elif LOSS == "gram":
                loss = texture_loss(rgb, target_grams)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pool[idx] = states.detach() #maj du pool avec les nouveaux etats
            loss_history.append(loss.item())

            if step % 200 == 0:
                print(f"step {step} ; loss {loss.item()}")
                save_image(states[0,:3,:,:].clamp(0, 1), f"{OUT_DIR}/step_{step}.png") #enregistrer que rgb pour visualiser les images
        
        return nca, loss_history



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                    MULTI-TEXTURES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if MULTI_TEX:

    def make_pool(size=1024, C=C, H=128, W=128):

        pool = torch.rand(size, C, H, W)
        chunk_size = size // N_TEX

        # initialisation pour chaque texture des images de sa pool personnelle
        # pour les "marquer" comme étant images de cette texture
        for tex_idx in range(N_TEX):
            bin_code = idx_to_bin(tex_idx, N_G)
            for bit_idx, cha_idx in enumerate(GENOMIC_CHANNELS):
                pool[tex_idx*chunk_size:(tex_idx+1)*chunk_size, cha_idx, :, :] = bin_code[bit_idx]

        return pool

    def sample_pool(pool, batch_size, tex_idx):
        """
        """

        chunk_size = len(pool) // N_TEX
        start_idx, end_idx = tex_idx*chunk_size, (tex_idx+1)*chunk_size

        idx = random.sample(range(start_idx, end_idx), batch_size)
        batch = pool[idx].clone()
        batch[0] = torch.rand(batch[0].shape)   #jai choisi le premier au pif car aleatoire 
        bin_code = idx_to_bin(tex_idx, N_G)
        for bit_idx, cha_idx in enumerate(GENOMIC_CHANNELS):
            batch[0, cha_idx, :, :] = bin_code[bit_idx]

        return batch, idx


    def train(nca, list_targets, list_target_grams, steps, batch, H, W, device):
        """
        """

        # regarder texture_loss dans loss.py et mettre les bons parametres

        pool = make_pool().to(device)
        optimizer = optim.Adam(nca.parameters(), lr=2e-3)
        loss_history = []

        for step in tqdm(range(steps)):

            if step == 2000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 2e-4

            # on tire au sort une texture pour piocher dans sa pool
            tex_idx = random.randint(0, N_TEX-1)
            states, idx = sample_pool(pool, batch, tex_idx)

            n=random.randint(32, 128) 
            states = nca(states, steps=n)

            rgb = states[:, :3, :, :]

            if LOSS == "sot":
                loss = texture_loss_sot(rgb, list_targets[tex_idx])
            elif LOSS == "gram":
                loss = texture_loss(rgb, list_target_grams[tex_idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pool[idx] = states.detach() #maj du pool avec les nouveaux etats
            loss_history.append(loss.item())

            if step % 200 == 0:
                print(f"step {step} ; loss {loss.item()} ; texture id {tex_idx}")
                save_image(states[0,:3,:,:].clamp(0, 1), f"{OUT_DIR}/step_{step}_tex_{tex_idx}.png") #enregistrer que rgb pour visualiser les images
        
        return nca, loss_history

