import torch

# Fichier de gestion, qui permet de régler l'ensemble des hyperparamètres.

# Mettre "True" pour entrainer un modèle multi-texture, "False" pour le modèle standard.
MULTI_TEX = False

# Choix des images, qui se trouvent dans le dossier "textures" (seul le nom de l'image est à préciser)
if not MULTI_TEX: # Choix de l'image pour le modèle standard
    IMAGE_PATH = "mixte.PNG" # Nom de l'image source
    C = 12


if MULTI_TEX:

    N_G = 2 # On génére 2^N_G textures différentes, donc il faut mettre le bon nombre d'images après (ici 4).
    IMAGES_PATHS = [ 
        "ecailles.jpg", # Nom des images sources
        "brique.jpg",
        "cerise.jpg",
        "moquette.jpg"
    ]

    TEX_IDX = 2 # En phase d'inférence, le modèle génère une texture à la fois. Ici, il s'agit donc du paramètre qui détermine sur quelle texture bascule le modèle (placement dans le vecteur ci-dessus)
    C = 18 # C = 3 + N_G + N_H


    # Case fixe qui gère le code génétique pour le modèle multi-texture. Ne pas modifier.
    ############################################################################################

    if len(IMAGES_PATHS) != 2**N_G:
        raise ValueError(f"Le vecteur IMAGES_PATHS doit être de taille 2^N_G = {2**N_G}.") # Erreur si le modèle multi-texture n'a pas le bon nombre de sources
    
    N_TEX = len(IMAGES_PATHS) # Nombre de textures sources
    GENOMIC_CHANNELS = list(range(3, 3 + N_G)) # Canaux génétiques (pour le modèle multi-texture), correspondant aux N_G canaux qui codent la texture à générer

    def idx_to_bin(tex_idx, n_g):
        """
        convertit l'indice d'une texture en son code binaire ("code génétique")
        """
        return [float((tex_idx >> i) & 1) for i in range(n_g)]

    TARGET_TEX = idx_to_bin(TEX_IDX, N_G) # Code génétique de la texture à générer


###############################################################################################

OUT_DIR = "output" # Placement des images générées
STEPS = 10000 # Nombre de pas du modèle (en cas multi-texture, on estime STEPS/N_TEX pas par texture, donc il faut l'augmenter). Ces paramètres sont réglés de manière assez empirique en pratique.
BATCH = 4 # Taille du batch 
SIZE = 128 # Taille de l'image générée 
HIDDEN = 96 # Nombre de canaux du réseau 
P = 0.5 # Probabilité de dropout 


LAYERS = [1, 6, 11, 18] # Couches du réseau importé VGG16 qui sont utilisées dans la loss
MEAN   = [0.485, 0.456, 0.406] #moyenne imagenet
STD    = [0.229, 0.224, 0.225] #std imagenet

PRESET = 0 # Réglage des filtres. 0 correspond aux réglages standard de l'article. Il est possible de le changer pour tester les hypothèses de l'article (cf. README.md)
FILTER_SIZE = 3 # Taille des filtres (uniquement pour les presets 5 et 6)

LOSS = "gram"  # sot ou gram, si on utilise le "Sliced-Optimal Transport" (SOT) ou la loss de style classique (Gram matrix)

INFERENCE = True # True si on utilise le modèle (qui doit se trouver dans le dossier "output"), False si on entraine le modèle.
# /!\ si MULTI_TEX = True et INFERENCE = True, il faut préciser TEX_IDX pour savoir quelle texture générer.

NB_IMGS = 10 # Nombre d'images à générer (en cas d'inférence)


# Réglages selon l'ordinateur
# Sur Mac :
# device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
# Sinon :
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
