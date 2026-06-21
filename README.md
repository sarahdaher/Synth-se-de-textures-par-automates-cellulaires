# Synthèse de textures par automates cellulaires neuronaux

**Héloïse Gouarné, Sarah Langlois, Baptiste Lefebvre et Sarah Daher**

Implémentation PyTorch from scratch d'un NCA pour la synthèse de textures (Mordvintsev et al., 2021), avec extension multi-texture (Catrina et al.) et Sliced Optimal Transport. Pour les détails théoriques, se référer au rapport.

---

## Installation

```bash
pip install torch torchvision tqdm Pillow
```

---

## Structure

```
main.py        
nca.py         # architecture NCA
train.py       # boucle d'entraînement
loss.py        # Gram matrices et SOT
config.py      # tous les hyperparamètres
utils.py       # chargement/sauvegarde, apply_damage
textures/      # images sources
output/        # images générées + modèles sauvegardés
```

---

## Utilisation

Tout se configure dans `config.py`, puis :

```bash
python main.py
```

**Entraînement** : `INFERENCE = False` -> modèle sauvegardé dans `output/nca.pth` (ou `nca_mult.pth`), courbe de loss dans `output/loss_history.png`.

**Inférence** : `INFERENCE = True` -> génère `NB_IMGS` images dans `output/preset_{PRESET}`.

**Test de reconstruction après dommage** : décommenter le bloc `apply_damage` dans `main.py`.

---

## Paramètres (`config.py`)

| Paramètre | Rôle | options |
|-----------|------|:------:|
| `MULTI_TEX` | Mode multi-texture | `False` ou `True` |
| `IMAGE_PATH` | Image source (dans `textures/`) | "path de l'image" |
| `C` | Canaux d'état par cellule | défaut : `12` |
| `HIDDEN` | Taille couche cachée MLP | défaut :`96` |
| `P` | Probabilité de mise à jour | défaut : `0.5` |
| `STEPS` | Steps d'entraînement | nb, défaut :`10000` |
| `BATCH` | Taille du batch | défaut :`4` |
| `SIZE` | Résolution  | défaut :`128` |
| `PRESET` | Filtres de perception (0–7) | `0`, `1`, ..., `7` |
| `LOSS` | Fonction coût choisie | `"gram"` ou `"sot"` |
| `INFERENCE` | Inférence vs entraînement | `True` ou `False` |
| `NB_IMGS` | nb d'images générées en inférence | nb, défaut : 10 |

### Mode multi-texture (`MULTI_TEX = True`)

| Paramètre | Rôle |
|-----------|------|
| `N_G` | Bits du code génomique (génère `2^N_G` textures) |
| `IMAGES_PATHS` | Liste des `2^N_G` images sources |
| `TEX_IDX` | Index de la texture à générer en inférence |
| `C` | Passer à `18` (= 3 RGB + N_G + canaux cachés) |

---

## Presets de filtres

| Preset | Filtres |
|--------|---------|
| 0 | ceux du papier : I + Sobel X/Y + Laplacien |
| 1 | I + Prewitt X/Y + Laplacien |
| 2 | I + Scharr X/Y + Laplacien |
| 3 | Gaussien + Sobel X/Y + Laplacien |
| 4 | I + 3 filtres aléatoires positifs |
| 5 | I + 3 filtres aléatoires de moyenne nulle |
| 6 | 4 filtres aléatoires de moyenne nulle  |
| 7 | I  |

---

## Références

-  Mordvintsev, Niklasson, Randazzo., *Texture Generation with Neural Cellular Automata*,  https://arxiv.org/abs/2105.07299
- Catrina, Plajer, Băicoianu, *Multi-texture synthesis through signal responsive neural cellular automata*, https://doi.org/10.1038/s41598-025-23997-7
- Gatys et al., *A Neural Algorithm of Artistic Style*, 2015, https://arxiv.org/pdf/1508.06576
- Peyré, Gabriel. *Sliced Optimal Transport Matching*, 2023. https://github.com/gpeyre/numerical-tours/blob/master/matlab/optimaltransp_4_matching_sliced.ipynb
- Code loss VGG : https://storimaging.github.io/notebooksImageGeneration/
