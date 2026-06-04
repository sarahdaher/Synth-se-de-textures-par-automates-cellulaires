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


if not MULTI_TEX:

    target = load_texture(IMAGE_PATH, size=SIZE).to(device)

    target_grams = get_target_grams(target)
    """
    #commenté l'ancienne version (réu du 27/06)
    for preset_filter in range(7):

        nca = NCA(C=C, hidden=HIDDEN, p=P, preset=preset_filter).to(device)

        if INFERENCE:
            nca.load_state_dict(torch.load(f"{OUT_DIR}/nca.pth"))
        else:
            nca = train(nca, target_grams,steps=STEPS, batch=BATCH, H=SIZE, W=SIZE, device=device)

        torch.save(nca.state_dict(), f"{OUT_DIR}/nca.pth")

        nca.eval()
        with torch.no_grad():
            state    = torch.rand(NB_IMGS, C, SIZE, SIZE, device=device)
            state    = nca(state, steps=200)
            #damaged  = apply_damage(state, radius=0.2)
            #recons = nca(damaged, steps=200)

        for i in range(NB_IMGS):
            os.makedirs(f"{OUT_DIR}/preset_{preset_filter}", exist_ok=True)
            save_image(state[i,:3,:,:], f"{OUT_DIR}/preset_{preset_filter}/final_{i}.png")
    #save_image(state[0,:3, :, :],    f"{OUT_DIR}/final.png")
    #save_image(damaged[:,:3],  f"{OUT_DIR}/damaged.png")
    #save_image(recons[:,:3], f"{OUT_DIR}/reconstruted.png")
    """

    if not INFERENCE:
        nca = NCA(C=C, hidden=HIDDEN, p=P, preset=PRESET).to(device)
        nca, loss_history = train(nca, target_grams, steps=STEPS, batch=BATCH, H=SIZE, W=SIZE, device=device)
        torch.save(nca.state_dict(), f"{OUT_DIR}/nca.pth")
        plt.plot(loss_history)
        plt.yscale("log")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title(f"Training Loss for {IMAGE_PATH.split('/')[-1]} ")
        plt.savefig(f"{OUT_DIR}/loss_history.png")
        plt.show()
    else:
        nca = NCA(C=C, hidden=HIDDEN, p=P, preset=PRESET).to(device)
        nca.load_state_dict(torch.load(f"{OUT_DIR}/nca.pth"))
        nca.eval()
        with torch.no_grad():
            state= torch.rand(NB_IMGS, C, SIZE, SIZE, device=device)
            state= nca(state, steps=200)

        for i in range(NB_IMGS):
            os.makedirs(f"{OUT_DIR}/preset_{PRESET}", exist_ok=True)
            save_image(state[i,:3,:,:].clamp(0, 1), f"{OUT_DIR}/preset_{PRESET}/final_{i}.png")


if MULTI_TEX:

    list_target_grams = []
    for path in IMAGES_PATHS:
        target = load_texture(path, size=SIZE).to(device)
        list_target_grams.append(get_target_grams(target))   

    if not INFERENCE:
        nca = NCA(C=C, hidden=HIDDEN, p=P, preset=PRESET).to(device)
        nca, loss_history = train(nca, list_target_grams, steps=STEPS, batch=BATCH, H=SIZE, W=SIZE, device=device)
        torch.save(nca.state_dict(), f"{OUT_DIR}/nca_mult.pth")
        plt.plot(loss_history)
        plt.yscale("log")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.savefig(f"{OUT_DIR}/loss_history.png")
        plt.show()
    else:
        nca = NCA(C=C, hidden=HIDDEN, p=P, preset=PRESET).to(device)
        nca.load_state_dict(torch.load(f"{OUT_DIR}/nca_mult.pth"))
        nca.eval()
        with torch.no_grad():
            state= torch.rand(NB_IMGS, C, SIZE, SIZE, device=device)

            # on initialise les genomic channels pr dire "on veut synthétiser CETTE texture"
            for bit_idx, cha_idx in enumerate(GENOMIC_CHANNELS):
                state[:, cha_idx, :, :] = TARGET_TEX[bit_idx]
        
            state= nca(state, steps=200)

        # pr le nom de dossier d'output du test
        filename = os.path.basename(IMAGES_PATHS[TEX_IDX])
        tex_name = texture_name = os.path.splitext(filename)[0]

        for i in range(NB_IMGS):
            os.makedirs(f"{OUT_DIR}/preset_{PRESET}", exist_ok=True)
            save_image(state[i,:3,:,:].clamp(0, 1), f"{OUT_DIR}/tex_{tex_name}_preset_{PRESET}/final_{i}.png")
