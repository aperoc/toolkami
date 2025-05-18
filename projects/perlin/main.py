#!/usr/bin/env -S PYTHONPATH=. uv run --script
# /// script
# dependencies = [ "numpy", "matplotlib", "scikit-image"]
# ///


import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

########### PERLIN + FRACTAL NOISE ###########
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

########### TARGET IMAGE ###########
H,W = 128,128
target_img = np.array(Image.open("/workspaces/toolkami/projects/perlin/fire.jpg").convert("L").resize((W,H))) / 255.0

########### FITNESS FUNCTION ###########
def mse(a,b): return np.mean((a-b)**2)
def fitness(noise, target, use_ssim=False):
    if use_ssim:
        return ssim(target, noise)                 # maximise SSIM
    return -mse(target, noise)                     # maximise (negative MSE)

########### GENERATE AND SCORE SINGLE SAMPLE ###########
base_res = 8  # base resolution
octaves = 4   # number of octaves
persistence = 0.5  # persistence value

# Generate a single fractal noise image
noise_img = fractal((H,W), (base_res,base_res), octaves, persistence)

# Calculate and print the score
score = fitness(noise_img, target_img)
print(f"Score: {score:.4f}")
