import torch

MULTI_TEX = False


if not MULTI_TEX:
    IMAGE_PATH = "textures/bubbles.png"
    C = 12


if MULTI_TEX:

    N_G = 2 # number of genomic channels !!! 2^N_G = N_TEX textures générées
    IMAGES_PATHS = [ # Le vecteur doit être de taille 2^N_G ! --- indexes 0, 1, …, N_TEX
        "ecailles.jpg",
        "brique.jpg",
        "cerise.jpg",
        "moquette.jpg"
    ]
    TEX_IDX = 2 # index de la texture que l'on souhaite générer pour le TEST
    C = 18 # C = 3 + N_G + N_H


    # PAS BESOIN DE S'EN PREOCCUPER ICI, MACROS CALCULEES TT SEULES
    # -------------------------------------------------------------

    if len(IMAGES_PATHS) != 2**N_G:
        raise ValueError(f"Le vecteur IMAGES_PATHS doit être de taille 2^N_G = {2**N_G}.")
    
    N_TEX = len(IMAGES_PATHS)
    GENOMIC_CHANNELS = list(range(3, 3 + N_G))

    def idx_to_bin(tex_idx, n_g):
        """
        convertit l'indice d'une texture en son code binaire ("code génétique")
        """
        return [float((tex_idx >> i) & 1) for i in range(n_g)]

    TARGET_TEX = idx_to_bin(TEX_IDX, N_G)


###############################################################################################

OUT_DIR = "output"
STEPS = 10000
BATCH = 4
SIZE = 128
HIDDEN = 96
P = 0.5

LAYERS = [1, 6, 11, 18] 
MEAN   = [0.485, 0.456, 0.406] #moyenne imagenet
STD    = [0.229, 0.224, 0.225] #std imagenet

PRESET = 1

LOSS = "gram"  # sot ou gram

INFERENCE = False
# !!!! si multi-textures :
# si INFERENCE = True, ne pas oublier de préciser TEX_IDX la texture qu'on veut générer!!!

NB_IMGS = 10

# mac
# device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

#pas mac
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")