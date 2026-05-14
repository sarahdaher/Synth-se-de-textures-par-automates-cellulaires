import torch
import torchvision.models as tv
import torch.nn as nn
import torch.nn.functional as F
from config import *


# fonctions similaires à http://colab.research.google.com/github/storimaging/Notebooks/blob/main/ImageGeneration/Style_Transfer.ipynb#scrollTo=CHKjVMHpYgkd


def gram_matrix(tnsr):
    """
    f représente les activations d'une couche du VGG
    f de shape(B, C, H, W) et on return (B, C, C)
    regarder la ref 11 du papier pour la formule et le code est pruis de la ressource ci-dessus
    """
    b, c, h, w = tnsr.size()
    Fm = tnsr.view(b, c, h * w)
    G = torch.bmm(Fm, Fm.transpose(1, 2))
    G.div_(h * w)
    return G


#charge VGG16 : fonction a part car avant on chargait a chaque calcul
def get_vgg():
    vgg = tv.vgg16( weights=tv.models.VGG16_Weights.IMAGENET1K_V1).features.eval()

    for p in vgg.parameters():
        p.requires_grad = False #bloque psk on utilise préentrainé

    return vgg


VGG = get_vgg()



def get_target_grams(img): #sera appele dans main pr calculer les target_grams

    grams = []
    x = img
    for i, layer in enumerate(VGG):
        x = layer(x)
        if i in LAYERS:
            grams.append(gram_matrix(x))
    return grams


# texture loss
def texture_loss(y_pred, target_grams, weights=[1., 1., 1., 1.]): #sera appele dans train.py pr calculer la loss
    """
    Calcule la loss entre pred_rgb et target_grams et la retourner

    y_pred: (B, 3, H, W) sortie RGB du NCA en [0,1] (considerer qu'on a deja pris les 3 canaux dans train.py donc qu'on a la bonne entree)
    target_grams: liste de 4 matrices de Gram (precalculees)
    weight: poids pour chaque couche
    appliquer eqt 2 de la ref 11 du papier 
    
    SOURCE DU HAUT
    """
    pred_grams = get_target_grams(y_pred)
    loss = 0
    for G, A, w in zip(pred_grams, target_grams, weights):

        loss += w * F.mse_loss(G, A)

    return loss
