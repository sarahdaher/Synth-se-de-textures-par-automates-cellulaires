import random
import torch
import torch.optim as optim
from config import *
from loss import texture_loss, texture_loss_sot
from utils import save_image
from tqdm import tqdm


# Séparation selon le modèle

if not MULTI_TEX:

    def make_pool(size=1024, C=12, H=SIZE, W=SIZE):
        """
        Crée un pool de size états initiaux avec du bruit U[0,1]  
        Renvoie un tenseur (size, C, H, W)
        """
        return torch.rand(size, C, H, W)


    def sample_pool(pool, batch_size):
        """
        Pioche batch_size états aléatoires dans le pool 
        Force la réinitialisation du premier état tiré (aléatoirement) avec du bruit aléatoire, comme décrit dans le papier :
        "we replace one of them with an empty state, so the model doesn't forget how to build the pattern from scratch"
        
        Renvoie batch (batch_size, C, H, W), idx (liste d'indices) 
        """
        
        idx      = random.sample(range(len(pool)), batch_size) # Choix aléatoire d'indices
        batch    = pool[idx].clone() # On clone pour ne pas modifier le pool directement
        batch[0] = torch.rand(batch[0].shape)   # On prend le premier car l'aléatoire est induit par radom.sample au-dessus
        return batch, idx
    
    
    def train(nca, target, target_grams, steps, batch, H, W, device):
        """
        Boucle d'entraînement  

        lr : 2e-3 jusqu'au step 2000, puis 2e-4 (cf. article)
        """

        # regarder texture_loss dans loss.py et mettre les bons paramètres

        # Initialisation du pool, qui est un ensemble d'états (images) que le modèle va utiliser pour s'entraîner
        pool = make_pool().to(device)
        optimizer = optim.Adam(nca.parameters(), lr=2e-3)
        loss_history = []

        for step in tqdm(range(steps)): # Boucle d'entraînement - assez standard, sauf quelques détails précisés

            if step == 2000: # Changement décrit par l'article
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 2e-4

            states, idx = sample_pool(pool, batch) # Pioche décrite ci-dessus
            n=random.randint(32, 64) 
            states = nca(states, steps=n) # Application du modèle

            rgb = states[:, :3, :, :] # On ne prend que les 3 premiers canaux, qui correspondent à l'image RGB générée par le modèle

            # Switch loss selon config
            if LOSS == "sot":
                loss = texture_loss_sot(rgb, target) # cf loss.py pour la fonction texture_loss_sot
            elif LOSS == "gram":
                loss = texture_loss(rgb, target_grams) # idem

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pool[idx] = states.detach() # Mise à jour du pool avec les nouveaux états
            loss_history.append(loss.item())

            if step % 200 == 0: # Visualisation pour vérifier que tout va bien !
                print(f"step {step} ; loss {loss.item()}")
                save_image(states[0,:3,:,:].clamp(0, 1), f"{OUT_DIR}/step_{step}.png") # N'enregistre que les canaux RGB pour visualiser les images
        
        return nca, loss_history

# Attention, ici on passe au modèle multi-texture

if MULTI_TEX:

    def make_pool(size=1024, C=C, H=SIZE, W=SIZE):

        """
        Crée un pool de size états initiaux avec du bruit U[0,1]  
        Renvoie un tenseur (size, C, H, W)  
        Chaque texture a sa propre pool, donc on "marque" les images de chaque pool avec le code génétique correspondant à la texture pour ne pas tout mélanger.
        """

        pool = torch.rand(size, C, H, W)
        chunk_size = size // N_TEX # Division de la pool selon le nombre de textures cibles

        # Initialisation pour chaque texture des images de sa "pool personnelle"
        # pour les "marquer" comme étant images de cette texture
        for tex_idx in range(N_TEX):
            bin_code = idx_to_bin(tex_idx, N_G)
            for bit_idx, cha_idx in enumerate(GENOMIC_CHANNELS):
                pool[tex_idx*chunk_size:(tex_idx+1)*chunk_size, cha_idx, :, :] = bin_code[bit_idx] # Marquage des images (initialisation des canaux génomiques)

        return pool

    def sample_pool(pool, batch_size, tex_idx):
        """
        Tire un échantillon de la pool de la texture spécifiée.
        """

        chunk_size = len(pool) // N_TEX
        start_idx, end_idx = tex_idx*chunk_size, (tex_idx+1)*chunk_size
        # Ici, c'est comme pour le modèle standard
        idx = random.sample(range(start_idx, end_idx), batch_size)
        batch = pool[idx].clone()
        batch[0] = torch.rand(batch[0].shape)
        # On doit bien marquer sur quelle texture on travaille
        bin_code = idx_to_bin(tex_idx, N_G)
        for bit_idx, cha_idx in enumerate(GENOMIC_CHANNELS):
            batch[0, cha_idx, :, :] = bin_code[bit_idx] # Marquage de l'image réinitialisée pour qu'elle corresponde à la texture choisie

        return batch, idx


    def train(nca, list_targets, list_target_grams, steps, batch, H, W, device):
        """
        Boucle d'entraînement pour le modèle multi-texture, essentiellement identique à l'entraînement standard. On ajoute juste un tirage au sort d'une texture à chaque étape pour piocher dans sa pool correspondante.
        """

        # Regarder texture_loss dans loss.py et mettre les bons paramètres

        pool = make_pool().to(device)
        optimizer = optim.Adam(nca.parameters(), lr=2e-3)
        loss_history = []

        for step in tqdm(range(steps)):

            if step == 2000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 2e-4

            # On tire au sort une texture pour piocher dans sa pool, seule difference avec le modèle standard
            tex_idx = random.randint(0, N_TEX-1)
            states, idx = sample_pool(pool, batch, tex_idx)

            n=random.randint(32, 64) 
            states = nca(states, steps=n)

            rgb = states[:, :3, :, :]

            if LOSS == "sot":
                loss = texture_loss_sot(rgb, list_targets[tex_idx])
            elif LOSS == "gram":
                loss = texture_loss(rgb, list_target_grams[tex_idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pool[idx] = states.detach() 
            loss_history.append(loss.item())

            if step % 200 == 0:
                print(f"step {step} ; loss {loss.item()} ; texture id {tex_idx}")
                save_image(states[0,:3,:,:].clamp(0, 1), f"{OUT_DIR}/step_{step}_tex_{tex_idx}.png") 
        
        return nca, loss_history

