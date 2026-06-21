import torch
import torchvision.models as tv
import torch.nn as nn
import torch.nn.functional as F
from config import *
from utils import *


# fonctions similaires à http://colab.research.google.com/github/storimaging/Notebooks/blob/main/ImageGeneration/Style_Transfer.ipynb#scrollTo=CHKjVMHpYgkd

def normalize(x): 
    """normalisation ImageNet"""
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor(STD,  device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def gram_matrix(tnsr):
    """
    Calcule la matrice de Gram d'un tenseur de features
    tnsr représente les activations d'une couche du VGG
    tnsr de shape(B, C, H, W) et on return (B, C, C)
    regarder la ref 11 du papier pour la formule et le code est pris de la ressource ci-dessus
    """
    b, c, h, w = tnsr.size()
    Fm = tnsr.view(b, c, h * w)
    G = torch.bmm(Fm, Fm.transpose(1, 2))
    G /= (h * w)
    return G


#charge VGG16 
def get_vgg():
    vgg = tv.vgg16(weights=tv.VGG16_Weights.IMAGENET1K_V1).features.eval()
    for p in vgg.parameters():
        p.requires_grad = False #bloque psk on utilise préentrainé

    return vgg


VGG = get_vgg().to(device)


def get_target_grams(img):
    """matrices de Gram de l'image cible (texture de référence)"""
    grams = []
    x = normalize(img)
    for i, layer in enumerate(VGG):
        x = layer(x)
        if i in LAYERS:
            grams.append(gram_matrix(x).detach())
    return grams


def texture_loss(y_pred, target_grams, weights=[1/(64**2), 1/(128**2), 1/(256**2), 1/(512**2)]):
    """Calcule la perte entre l'image générée et la cible"""
    pred_grams = []
    x = normalize(y_pred)
    for i, layer in enumerate(VGG):
        x = layer(x)
        if i in LAYERS:
            pred_grams.append(gram_matrix(x))

    loss = 0
    for G, A, w in zip(pred_grams, target_grams, weights):
        loss += w * ((G - A.expand_as(G)) ** 2).mean()

    return loss


##### SOT : alternative aux matrices de Gram #####

def calc_styles_vgg(imgs):
    """extrait les features + rgb de l'image générée et de la cible pour calculer la loss SOT"""
    b, c, h, w = imgs.shape
    x = normalize(imgs.clamp(0, 1))
    features = [x.reshape(b, c, h * w)] #init avec rgb

    for i, layer in enumerate(VGG):
        x = layer(x)
        if i in LAYERS:
            b2, c2, h2, w2 = x.shape
            features.append(x.reshape(b2, c2, h2 * w2))

    return features


def project_sort(x, proj):
    """Projette les features sur des directions aléatoires puis trie les valeurs en 1D"""
    # x:(B, C, N), N=H*W, proj:(C, proj_n)
    #(B,C,N) -> (B,N,C) @ (C,P) -> (B,N,P) -> (B,P,N)
    return (x.permute(0, 2, 1) @ proj).permute(0, 2, 1).sort(dim=-1)[0]

def ot_loss(source, target, proj_n=32):
    """
    Perte  entre deux distributions de features via la méthode Sliced Wasserstein
    Principe : on projette source et target sur proj_n directions aléatoires, on trie chaque projection, 
    puis on mesure l'écart**2 (la distance de Wasserstein 1D entre deux distributions triées est exactement le L2 entre leurs versions triées)."""
    ch = source.shape[1]

    projs = torch.randn(ch, proj_n, device=source.device)
    projs = projs / (projs.norm(dim=0, keepdim=True) + 1e-8)

    source_proj = project_sort(source, projs)           # (B, P, N)
    target_proj = project_sort(target, projs)           # (1, P, N)
    target_proj = target_proj.expand_as(source_proj)    # (B, P, N)

    return (source_proj - target_proj).square().sum()

def texture_loss_sot(y_pred, target_img):
    pred_styles   = calc_styles_vgg(y_pred)
    target_styles = calc_styles_vgg(target_img)

    #detach la cible car pas de gradient sur l'image de ref
    loss = sum(ot_loss(p, t.detach()) for p, t in zip(pred_styles, target_styles))
    return loss