import os
import torch
import matplotlib.pyplot as plt

from nca    import NCA
from loss   import *
from train  import *
from utils  import *
from config import *


print(device)

os.makedirs(OUT_DIR, exist_ok=True)

# Main est séparé en deux, selon le modèle qui est utilisé ou entrainé (uni/multi-texture). 

if not MULTI_TEX:

    target = load_texture(IMAGE_PATH, size=SIZE).to(device) # Image source

    target_grams = get_target_grams(target) # Matrice de Gram de l'image source

    if not INFERENCE: # Boucle d'entrainement
        nca = NCA(C=C, hidden=HIDDEN, p=P, preset=PRESET).to(device)
        nca, loss_history = train(nca, target, target_grams, steps=STEPS, batch=BATCH, H=SIZE, W=SIZE, device=device) #cf fichier train.py pour la fonction train
        torch.save(nca.state_dict(), f"{OUT_DIR}/nca.pth")
        # Plot de la fonction de loss
        plt.plot(loss_history)
        plt.yscale("log")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title(f"Training Loss for {IMAGE_PATH.split('/')[-1]} ")
        plt.savefig(f"{OUT_DIR}/loss_history.png")
        plt.show()
    else: # Application du modèle
        nca = NCA(C=C, hidden=HIDDEN, p=P, preset=PRESET).to(device)
        nca.load_state_dict(torch.load(f"{OUT_DIR}/nca.pth"))
        
        nca.eval()
        
        with torch.no_grad():
            state= torch.rand(NB_IMGS, C, SIZE, SIZE, device=device)
            state= nca(state, steps=200)

        for i in range(NB_IMGS):
            os.makedirs(f"{OUT_DIR}/preset_{PRESET}", exist_ok=True)
            save_image(state[i,:3,:,:].clamp(0, 1), f"{OUT_DIR}/preset_{PRESET}/final_{i}.png")
        nca.eval()

        # si on veut appliquer damage, décommenter cette partie
        """
        with torch.no_grad():
            state   = torch.rand(1, C, SIZE, SIZE, device=device)
            state   = nca(state, steps=200)
            save_image(state[0, :3, :, :], f"{OUT_DIR}/original.png")

            damaged = apply_damage(state, rayon=0.2)
            save_image(damaged[0, :3, :, :], f"{OUT_DIR}/damaged.png")

            current = damaged.clone()
            for i in [10, 25, 50, 75, 100, 150, 200]:
                current = damaged.clone()  
                current = nca(current, steps=i)
                save_image(current[0, :3, :, :].clamp(0, 1), f"{OUT_DIR}/recons_step_{i}.png")
        """

if MULTI_TEX: # Presque la même chose

    list_targets = []
    list_target_grams = []
    for path in IMAGES_PATHS: # Images sources et matrices de Gram correspondantes
        t = load_texture(path, size=SIZE).to(device)
        list_targets.append(t)
        list_target_grams.append(get_target_grams(t))

    if not INFERENCE: # Boucle d'entrainement
        nca = NCA(C=C, hidden=HIDDEN, p=P, preset=PRESET).to(device)
        nca, loss_history = train(nca, list_targets, list_target_grams, steps=STEPS, batch=BATCH, H=SIZE, W=SIZE, device=device)
        torch.save(nca.state_dict(), f"{OUT_DIR}/nca_mult.pth")
        plt.plot(loss_history)
        plt.yscale("log")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.savefig(f"{OUT_DIR}/loss_history.png")
        plt.show()
    else: # Application du modèle
        nca = NCA(C=C, hidden=HIDDEN, p=P, preset=PRESET).to(device)
        nca.load_state_dict(torch.load(f"{OUT_DIR}/nca_mult.pth"))
        nca.eval()
        with torch.no_grad():
            state= torch.rand(NB_IMGS, C, SIZE, SIZE, device=device)

            # Initialisation des genomic channels 
            for bit_idx, cha_idx in enumerate(GENOMIC_CHANNELS):
                state[:, cha_idx, :, :] = TARGET_TEX[bit_idx]
        
            state= nca(state, steps=200)

        filename = os.path.basename(IMAGES_PATHS[TEX_IDX])
        tex_name = texture_name = os.path.splitext(filename)[0]

        for i in range(NB_IMGS):
            os.makedirs(f"{OUT_DIR}/tex_{tex_name}_preset_{PRESET}", exist_ok=True)
            save_image(state[i,:3,:,:].clamp(0, 1), f"{OUT_DIR}/tex_{tex_name}_preset_{PRESET}/final_{i}.png")
