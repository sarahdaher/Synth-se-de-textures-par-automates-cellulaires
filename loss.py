import torch
import torchvision.models as tv
import torch.nn as nn
import torch.nn.functional as F
from config import *

#layers qui nous interessent: indices 1, 6, 11, 18 de vgg.features (cf le papier)
#normalisation ImageNet : mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]
#regarder code du challenge ou on faisait du transfer learning pour des inspis c la meme idee

def gram_matrix(f):
    """
    f de shape(B, C, H, W) et on return (B, C, C)
    regarder la ref 11 du papier pour la formule (si jai bien compris c'est F*F.T et F est f reshapé en (B, C,HW) pour avoir les memes tailles que le papier, i et j canaux, k pixel dans l'eqt 1)
    """
    # TODO
    raise NotImplementedError


def get_target_grams(img):    #sera appele dans main pr calculer les target_grams
    """
    Charge VGG16 pre entraine, le geler, passer img dedans,
    retourne les matrices de Gram aux couches relu1,2,3,4 

    img taille (1, 3, H, W) en [0, 1], on return target_grams (liste de 4 Gram)
    """
    # TODO
    raise NotImplementedError


def texture_loss(pred, vgg, target_grams): #sera appele dans train.py pr calculer la loss
    """
    Calcule la loss entre pred_rgb et target_grams et la retourner

    pred: (B, 3, H, W) sortie RGB du NCA en [0,1] (considerer qu'on a deja pris les 3 canaux dans train.py donc qu'on a la bonne entree)
    target_grams: liste de 4 matrices de Gram (precalculees)
    
    appliquer eqt 2 de la ref 11 
    """
    # TODO
    raise NotImplementedError
