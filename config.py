import torch

IMAGE_PATH = "textures/bubbles.png"
OUT_DIR = "output"
STEPS = 5000
BATCH = 4
SIZE = 128
C = 12
HIDDEN = 96
P = 0.5

LAYERS = {1, 6, 11, 18}
MEAN   = [0.485, 0.456, 0.406] #moyenne imagenet
STD    = [0.229, 0.224, 0.225] #std imagenet

PRESET = 0

INFERENCE = True
NB_IMGS = 10

# mac
#device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

#pas mac
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")