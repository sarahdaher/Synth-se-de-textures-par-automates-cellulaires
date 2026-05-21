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


def scale(x):
    return (x - x.min()) / (x.max() - x.min())    


def gram_matrix(tnsr):
    """
    f représente les activations d'une couche du VGG
    f de shape(B, C, H, W) et on return (B, C, C)
    regarder la ref 11 du papier pour la formule et le code est pruis de la ressource ci-dessus
    """
    b, c, h, w = tnsr.size()
    Fm = tnsr.view(b, c, h * w)
    G = torch.bmm(Fm, Fm.transpose(1, 2))
    #G.div_(h * w)
    return G


#charge VGG16 : fonction a part car avant on chargait a chaque calcul
def get_vgg():
    vgg = tv.vgg16(weights=tv.VGG16_Weights.IMAGENET1K_V1).features.eval()
    for p in vgg.parameters():
        p.requires_grad = False #bloque psk on utilise préentrainé

    return vgg


VGG = get_vgg().to(device) 


def get_target_grams(img): #sera appele dans main pr calculer les target_grams
    grams = []
    x = normalize(img)
    for i, layer in enumerate(VGG):
        x = layer(x)
        if i in LAYERS:
            grams.append(gram_matrix(x).detach())  # FIX 1 : detach pour pas garder le graphe VGG
    return grams


def texture_loss(y_pred, target_grams, weights=[1., 1., 1., 1.]):
    pred_grams = []
    x = normalize(y_pred.clamp(0, 1))
    for i, layer in enumerate(VGG):
        x = layer(x)
        if i in LAYERS:
            pred_grams.append(gram_matrix(x))  # pas de .detach() ici !

    loss = 0
    for G, A, w in zip(pred_grams, target_grams, weights):
        loss += w * ((G - A.expand_as(G)) ** 2).mean()

    return loss