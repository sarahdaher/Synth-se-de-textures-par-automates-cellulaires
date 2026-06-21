import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from config import *

# Fichier utiliser pour coder les fonctions "utilitaires", chargement, sauvegarde...


def load_texture(path, size=128): 
    """
    charge une image depuis path, la redimensionne en carré de côté size 
    return tensor (1, 3, H, W) en  [0,1] 
    """
    img = Image.open(path).convert('RGB') 
    img = img.resize((size, size))
    tensor_image = transforms.ToTensor()(img) # la normalisation est déjà faite
    tensor_image = tensor_image.unsqueeze(0) # pour ajouter une dimension batch (1, 3, H, W)
    return tensor_image


def save_image(tensor_image, path):
    """save l'image generee par nca"""
    pil_image = F.to_pil_image(tensor_image.clamp(0, 1)) # clamp par précaution, pour éviter les valeurs négatives ou >1
    pil_image.save(path)


def apply_damage(state, rayon=0.2):
    """efface un cercle de rayon rayon*min(H,W) au centre de l'image"""
    B, C, H, W = state.shape
    state = state.clone()
    cy, cx = H//2, W//2
    r = int(rayon * min(H, W))
    for y in range(H):
        for x in range(W):
            if (x-cx)**2 + (y-cy)**2 < r**2:
                state[:, :, y, x] = torch.rand(B, C, device=state.device)
    return state
