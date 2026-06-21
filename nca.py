import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

def make_kernels(C, preset=0):
    """Retourne un tensor avec les 4 noyaux (s, sx, sy, lap) et les concatener
    Chaque preset définit une famille de filtres différente
    0 ceux du papier de reference"""
    if preset == 0:
        s = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        sx  = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) /8
        sy  = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32) /8
        lap = torch.tensor([[1, 2, 1], [2, -12, 2], [1, 2, 1]], dtype=torch.float32) /24
        kernels = torch.stack([s, sx, sy, lap])

    elif preset == 1:
        s = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        sx  = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=torch.float32) /6
        sy  = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=torch.float32) /6
        lap = torch.tensor([[1, 2, 1], [2, -12, 2], [1, 2, 1]], dtype=torch.float32) /24
        kernels = torch.stack([s, sx, sy, lap])

    elif preset == 2:
        s = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        sx  = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32) /32
        sy  = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32) /32
        lap = torch.tensor([[1, 2, 1], [2, -12, 2], [1, 2, 1]], dtype=torch.float32) /24
        kernels = torch.stack([s, sx, sy, lap])

    elif preset == 3:
        gaussian = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) /16
        sx  = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) /8
        sy  = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32) /8
        lap = torch.tensor([[1, 2, 1], [2, -12, 2], [1, 2, 1]], dtype=torch.float32) /24
        kernels = torch.stack([gaussian, sx, sy, lap])

    elif preset == 4:
        s = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        s1 = torch.randint(0, 100, (3, 3), dtype=torch.float32)
        s1.div_(torch.sum(s1))
        s2 = torch.randint(0, 100, (3, 3), dtype=torch.float32)
        s2.div_(torch.sum(s2))
        s3 = torch.randint(0, 100, (3, 3), dtype=torch.float32)
        s3.div_(torch.sum(s3))
        kernels = torch.stack([s, s1, s2, s3])

    elif preset == 5:
        s = torch.zeros((FILTER_SIZE, FILTER_SIZE), dtype=torch.float32)
        s[FILTER_SIZE//2, FILTER_SIZE//2] = 1.0
        s1 = torch.randint(0, 100, (FILTER_SIZE, FILTER_SIZE), dtype=torch.float32)
        s1.div_(torch.sum(s1)) 
        s1 = s1 - torch.mean(s1)
        s2 = torch.randint(0, 100, (FILTER_SIZE, FILTER_SIZE), dtype=torch.float32)
        s2.div_(torch.sum(s2))
        s2 = s2 - torch.mean(s2)
        s3 = torch.randint(0, 100, (FILTER_SIZE, FILTER_SIZE), dtype=torch.float32)
        s3.div_(torch.sum(s3))
        s3 = s3 - torch.mean(s3)
        kernels = torch.stack([s, s1, s2, s3])    

    elif preset == 6:
        s1 = torch.randint(0, 100, (FILTER_SIZE, FILTER_SIZE), dtype=torch.float32)
        s1.div_(torch.sum(s1))
        s1 = s1 - torch.mean(s1)
        s2 = torch.randint(0, 100, (FILTER_SIZE, FILTER_SIZE), dtype=torch.float32)
        s2.div_(torch.sum(s2))
        s2 = s2 - torch.mean(s2)
        s3 = torch.randint(0, 100, (FILTER_SIZE, FILTER_SIZE), dtype=torch.float32)
        s3.div_(torch.sum(s3))
        s3 = s3 - torch.mean(s3)
        s4 = torch.randint(0, 100, (FILTER_SIZE, FILTER_SIZE), dtype=torch.float32)
        s4.div_(torch.sum(s4))
        s4 = s4 - torch.mean(s4)
        kernels = torch.stack([s1, s2, s3, s4])    
        kernels = kernels / (kernels.norm() + 1e-6)

    elif preset == 7:
        s = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32) /2
        kernels = torch.stack([s])

    kernels = kernels.unsqueeze(1).repeat(C, 1, 1, 1) # (4*C, 1, 3, 3)

    return kernels
    


class NCA(nn.Module):

    def __init__(self, C=C, hidden=HIDDEN, p=P, preset=0):
        """
        C : nombre de canaux du vecteur d'état de chaque cellule (12 dans l'article)
        p : probabilité de mise à jour
        """
        super().__init__()
        self.C = C
        self.p = p

        if preset <= 6:
            self.nb_filters = 4
        else:
            self.nb_filters = 1

        #enregistrer les kernels (self.kernels) comme buffer (avec la fct register_buffer pour pas les apprendre car ils sont fixes (sobel et laplacien) ; une fonction pytorch)
        kernels = make_kernels(C, preset)
        self.register_buffer("kernels", kernels)
    
        #le mlp
        self.mlp = nn.Sequential(
            nn.Conv2d(self.nb_filters*C, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden, C, kernel_size=1),
        )
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias) 


    def perceive(self, state):
        """Applique les nb_filters filtres sur chaque canal de la grille.
        Topologie torique : le padding circulaire connecte les bords opposés.
        Conv depthwise (groups=C) : chaque canal est filtré indépendamment."""

        state = F.pad(state, (FILTER_SIZE//2, FILTER_SIZE//2, FILTER_SIZE//2, FILTER_SIZE//2), mode='circular') #torus topolgy 
        # Conv depthwise : on applique les memes kernels sur chaque canal de la grille
        perception_vector = F.conv2d(state, self.kernels.view(-1, 1, FILTER_SIZE, FILTER_SIZE), groups=self.C)

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
