import torch
import numpy as np
from PIL import Image
from config import *

def load_texture(path, size=128):
    """
    vharge une image depuis path, la redimensionne en carrede cote size 
    return tensor (1, 3, H, W) en  [0,1] 
    """
    # TODO
    raise NotImplementedError


def save_image(tensors, path):
    """save l'image generee par nca"""
    # TODO
    raise NotImplementedError


#pour la partie de fin tt ca  si on a envie de tester la reconstruction d'image apres damage
def apply_damage(state, rayon):
    """
    efface un cercle au centre de la grille en mettant du bruit a la place
    """
    # TODO
    raise NotImplementedError
