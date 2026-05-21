import os
import torch

from nca    import NCA
from loss   import *
from train  import *
from utils  import *
from config import *

INFERENCE = False
NB_IMGS = 10

os.makedirs(OUT_DIR, exist_ok=True)

target = load_texture(IMAGE_PATH, size=SIZE).to(device)

target_grams = get_target_grams(target)

nca = NCA(C=C, hidden=HIDDEN, p=P).to(device)

if INFERENCE:
    nca.load_state_dict(torch.load(f"{OUT_DIR}/nca.pth"))
else:
    nca = train(nca, target_grams,steps=STEPS, batch=BATCH, H=SIZE, W=SIZE, device=device)

torch.save(nca.state_dict(), f"{OUT_DIR}/nca.pth")

nca.eval()
with torch.no_grad():
    state    = torch.rand(NB_IMGS, C, SIZE, SIZE, device=device)
    state    = nca(state, steps=200)
    #damaged  = apply_damage(state, radius=0.2)
    #recons = nca(damaged, steps=200)

for i in range(NB_IMGS):
    save_image(state[i,:3,:,:], f"{OUT_DIR}/final_{i}.png")
#save_image(state[0,:3, :, :],    f"{OUT_DIR}/final.png")
#save_image(damaged[:,:3],  f"{OUT_DIR}/damaged.png")
#save_image(recons[:,:3], f"{OUT_DIR}/reconstruted.png")
