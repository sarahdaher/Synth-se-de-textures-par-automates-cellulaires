import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

def make_kernels(C):
    """Retourne un tensor avec les 4 noyaux (s, sx, sy, lap) et les concatener"""
    # TODO
    raise NotImplementedError


class NCA(nn.Module):

    def __init__(self, C=C, hidden=HIDDEN, p=P):
        super().__init__()
        self.C = C
        self.p = p
        
        #TODO
        #enregistrer les kernels (self.kernels) comme buffer (avec la fct register_buffer pour pas les apprendre car ils sont fixes (sobel et laplacien) ; une fonction pytorch)
        #ecrire le mlp
        
        raise NotImplementedError

    def perceive(self, state):
        """Applique les 4 filtres sur la grille"""
        # torus topology (a voir comment faire??) + conv depthwise

        raise NotImplementedError

    def forward(self, state, steps):
        """
        renvoie taille (B, C, H, W)
        a chaque step : perceive ->self.mlp -> masque Bernoulli(self.p) de shape (B, 1, H, W) -> maj de s
        """
        # TODO
        raise NotImplementedError
