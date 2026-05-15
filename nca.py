import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

def make_kernels(C):
    """Retourne un tensor avec les 4 noyaux (s, sx, sy, lap) et les concatener"""
    s = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sy = torch.tensor([[-1, -2, 1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    lap = torch.tensor([[1, 2, 1], [2, -12, 2], [1, 2, 1]], dtype=torch.float32)
    kernels = torch.stack([s, sx, sy, lap])
    kernels = kernels.repeat(C, 1, 1, 1)
    raise kernels


class NCA(nn.Module):

    def __init__(self, C=C, hidden=HIDDEN, p=P):
        """
        Parameters
        ----------
        C : int
            nombre de canaux du vecteur d'état de chaque cellule (12 dans l'article)
        p : float
            probabilité de mise à jour
        """
        super().__init__()
        self.C = C
        self.p = p
        
        #enregistrer les kernels (self.kernels) comme buffer (avec la fct register_buffer pour pas les apprendre car ils sont fixes (sobel et laplacien) ; une fonction pytorch)
        kernels = make_kernels(C)
        self.register_buffer("kernels", kernels)

        #ecrire le mlp
        self.mlp = nn.Sequential(
            nn.Conv2d(4*C, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden, C, kernel_size=1),
        )

    def perceive(self, state):
        """Applique les 4 filtres sur la grille"""
        # torus topology (a voir comment faire??) + conv depthwise

        # Conv depthwise : on applique les memes kernels sur chaque canal de la grille
        perception_vector = F.conv2d(state, self.kernels.view(-1, 1, 3, 3), padding=1, groups=self.C)
        return perception_vector # (B, 4*C, H, W)

    def forward(self, state, steps):
        """
        renvoie taille (B, C, H, W)
        a chaque step : perceive ->self.mlp -> masque Bernoulli(self.p) de shape (B, 1, H, W) -> maj de s
        """
        for _ in range(steps):
            perception_vector = self.perceive(state) # (B, 4*C, H, W)
            bernoulli_mask = torch.bernoulli(torch.full((state.shape[0], 1, state.shape[2], state.shape[3]), self.p, device=state.device)) # (B, 1, H, W)
            state = state + bernoulli_mask * self.mlp(perception_vector)
        return state
