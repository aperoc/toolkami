#!/usr/bin/env -S PYTHONPATH=. uv run --script
# /// script
# dependencies = [ "numpy", "matplotlib", "scikit-image"]
# ///


import numpy as np, matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim

########### PERLIN + FRACTAL NOISE (same core as before) ###########
def fade(t): return 6*t**5 - 15*t**4 + 10*t**3
def perlin2d(shape, res):
    H,W = shape; ry,rx = max(1,res[0]), max(1,res[1])
    y = np.linspace(0, ry, H, endpoint=False); x = np.linspace(0, rx, W, endpoint=False)
    YY, XX = np.meshgrid(y%1, x%1, indexing='ij')
    angles = 2*np.pi*np.random.rand(ry+1, rx+1)
    gx, gy = np.cos(angles), np.sin(angles)
    def dot(ix,iy):
        # Get the correct slice of gradients
        gx_slice = gx[iy:iy+ry, ix:ix+rx]
        gy_slice = gy[iy:iy+ry, ix:ix+rx]
        # Calculate exact repeat factors to match target dimensions
        h_rep = H // gx_slice.shape[0]
        w_rep = W // gx_slice.shape[1]
        # Ensure we have at least one repeat
        h_rep = max(1, h_rep)
        w_rep = max(1, w_rep)
        # Resize to match XX and YY dimensions exactly
        gx_resized = np.repeat(np.repeat(gx_slice, h_rep, axis=0), w_rep, axis=1)
        gy_resized = np.repeat(np.repeat(gy_slice, h_rep, axis=0), w_rep, axis=1)
        # Trim if we overshot
        if gx_resized.shape[0] > H:
            gx_resized = gx_resized[:H, :]
            gy_resized = gy_resized[:H, :]
        if gx_resized.shape[1] > W:
            gx_resized = gx_resized[:, :W]
            gy_resized = gy_resized[:, :W]
        return ((XX-ix)*gx_resized + (YY-iy)*gy_resized)
    n00 = dot(0,0); n10 = dot(1,0); n01 = dot(0,1); n11 = dot(1,1)
    u, v = fade(XX), fade(YY)
    lerp = lambda a,b,t: a*(1-t)+b*t
    return lerp(lerp(n00,n10,u), lerp(n01,n11,u), v)

def fractal(shape, base_res, octaves=4, persistence=0.5):
    img = np.zeros(shape); amp=1; freq=1
    for _ in range(octaves):
        img += amp*perlin2d(shape, (int(base_res[0]*freq), int(base_res[1]*freq)))
        amp *= persistence; freq *= 2
    img = (img - img.min())/(img.max()-img.min())   # normalise 0-1
    return img

########### TARGET IMAGE (fire.jpg) ###########
H,W = 128,128
target_img = np.array(Image.open("/workspaces/toolkami/projects/perlin/fire.jpg").convert("L").resize((W,H))) / 255.0

# >>> Replace with something like:
# target_img = np.array(Image.open("your_fire.jpg").convert("L").resize((W,H))) / 255.0
#################################################################

########### FITNESS FUNCTION (negative MSE or SSIM) ###########
def mse(a,b): return np.mean((a-b)**2)
def fitness(noise, target, use_ssim=False):
    if use_ssim:
        return ssim(target, noise)                 # maximise SSIM
    return -mse(target, noise)                     # maximise (negative MSE)

########### EVOLUTION LOOP (tiny, 6 individuals, 5 generations) ###########
pop = [ (np.random.choice([4,8,16]), np.random.randint(2,6), np.random.uniform(0.3,0.8))
        for _ in range(6) ]   # (base res, octaves, persistence)
best_imgs=[]

for gen in range(6):
    scored=[]
    for base,octv,pers in pop:
        img = fractal((H,W),(base,base),octv,pers)
        scored.append( (fitness(img,target_img), (base,octv,pers), img) )
    scored.sort(reverse=True)
    best=scored[0]; print(f"G{gen}: score={best[0]:.4f}, params={best[1]}")
    best_imgs.append(best[2])

    # selection + mutation
    parents=[x[1] for x in scored[:2]]
    new_pop=parents.copy()
    while len(new_pop)<6:
        base,octv,pers = parents[np.random.randint(2)]
        if np.random.rand()<0.5: base=np.random.choice([4,8,16])
        if np.random.rand()<0.5: octv=max(1,octv+np.random.choice([-1,1]))
        if np.random.rand()<0.5: pers=float(np.clip(pers+np.random.normal(0,0.05),0.3,0.9))
        new_pop.append((base,octv,pers))
    pop=new_pop

########### VISUALISE target and best of each generation ###########
fig,axs=plt.subplots(2,4,figsize=(10,5))
axs[0,0].imshow(target_img,cmap='inferno'); axs[0,0].set_title("Target"); axs[0,0].axis('off')
for i,img in enumerate(best_imgs):
    r,c=divmod(i+1,4); axs[r,c].imshow(img,cmap='inferno')
    axs[r,c].set_title(f"G{i}"); axs[r,c].axis('off')
plt.tight_layout()
plt.savefig('perlin_evolution.png', dpi=300, bbox_inches='tight')
plt.close()
