import torch
import torchvision.models as tv
import torch.nn as nn
import torch.nn.functional as F
from config import *
from utils import *


# fonctions similaires à http://colab.research.google.com/github/storimaging/Notebooks/blob/main/ImageGeneration/Style_Transfer.ipynb#scrollTo=CHKjVMHpYgkd

def normalize(x):
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor(STD,  device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def gram_matrix(tnsr):
    """
    f représente les activations d'une couche du VGG
    f de shape(B, C, H, W) et on return (B, C, C)
    regarder la ref 11 du papier pour la formule et le code est pruis de la ressource ci-dessus
    """
    b, c, h, w = tnsr.size()
    Fm = tnsr.view(b, c, h * w)
    G = torch.bmm(Fm, Fm.transpose(1, 2))
    G /= (h * w)
    return G


#charge VGG16 : fonction a part car avant on chargait a chaque calcul
def get_vgg():
    vgg = tv.vgg16(weights=tv.VGG16_Weights.IMAGENET1K_V1).features.eval()
    for p in vgg.parameters():
        p.requires_grad = False #bloque psk on utilise préentrainé

    return vgg


VGG = get_vgg().to(device)


def get_target_grams(img):
    grams = []
    x = normalize(img)
    for i, layer in enumerate(VGG):
        x = layer(x)
        if i in LAYERS:
            grams.append(gram_matrix(x).detach())
    return grams


def texture_loss(y_pred, target_grams, weights=[1/(64**2), 1/(128**2), 1/(256**2), 1/(512**2)]):
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


#####SOT#####
def calc_styles_vgg(imgs):
    b, c, h, w = imgs.shape
    x = normalize(imgs.clamp(0, 1))

    features = []
    for i, layer in enumerate(VGG):
        x = layer(x)
        if i in LAYERS:
            b2, c2, h2, w2 = x.shape
            features.append(x.reshape(b2, c2, h2*w2))

    return features


def project_sort(x, proj):
    # x:(B, C, N), N=H*W, proj:(C, proj_n)
    #(B,C,N) -> (B,N,C) @ (C,P) -> (B,N,P) -> (B,P,N)
    return (x.permute(0, 2, 1) @ proj).permute(0, 2, 1).sort(dim=-1)[0]


def ot_loss(source, target, proj_n=512):
    ch = source.shape[1]   
    n  = source.shape[2]   

    projs = torch.randn(ch, proj_n, device=source.device)
    projs = projs / (torch.norm(projs, dim=0, keepdim=True) + 1e-8)

    source_proj = project_sort(source, projs)   
    target_proj = project_sort(target, projs)   

    if target_proj.shape[-1] != n:
        target_proj = F.interpolate(target_proj, size=n, mode="nearest")

    return (source_proj - target_proj).square().mean()


def texture_loss_sot(y_pred, target_img, weights=[1, 1, 1, 1]):
    pred_styles   = calc_styles_vgg(y_pred)
    target_styles = calc_styles_vgg(target_img)

    loss = 0
    for pred, target, w in zip(pred_styles, target_styles, weights):
        loss = loss + w * ot_loss(pred, target.detach())

    return loss


def color_loss_sot(y_pred, target_img):
    loss = ot_loss(y_pred.view(y_pred.shape[0], y_pred.shape[1], -1), target_img.view(target_img.shape[0], target_img.shape[1], -1).detach(), proj_n=1024)
    return loss