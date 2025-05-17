#!/usr/bin/env -S PYTHONPATH=. uv run --script
# /// script
# dependencies = [ "numpy", "matplotlib", "scikit-image"]
# ///


import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

########### TARGET IMAGE ###########
H,W = 128,128
target_img = np.array(Image.open("/workspaces/toolkami/projects/perlin/fire.jpg").convert("L").resize((W,H))) / 255.0

########### FITNESS FUNCTION ###########
def mse(a,b): return np.mean((a-b)**2)
def fitness(noise, target, use_ssim=False):
    if use_ssim:
        return ssim(target, noise)                 # maximise SSIM
    return -mse(target, noise)                     # maximise (negative MSE)

# EVOLVE-BLOCK START
def fade(t): return 6*t**5 - 15*t**4 + 10*t**3
def perlin2d(shape, res):
    H, W = shape
    ry, rx = max(1, res[0]), max(1, res[1])
    
    # Create coordinate grids
    y = np.linspace(0, ry, H, endpoint=False)
    x = np.linspace(0, rx, W, endpoint=False)
    YY, XX = np.meshgrid(y % 1, x % 1, indexing='ij')
    
    # Generate random gradients
    angles = 2 * np.pi * np.random.rand(ry + 1, rx + 1)
    gx = np.cos(angles)
    gy = np.sin(angles)
    
    # Calculate dot products for all four corners at once
    def dot_product(ix, iy):
        # Get gradients for the current cell
        gx_cell = gx[iy:iy+ry, ix:ix+rx]
        gy_cell = gy[iy:iy+ry, ix:ix+rx]
        
        # Resize gradients to match target dimensions
        gx_resized = np.repeat(np.repeat(gx_cell, H // gx_cell.shape[0], axis=0), 
                             W // gx_cell.shape[1], axis=1)[:H, :W]
        gy_resized = np.repeat(np.repeat(gy_cell, H // gy_cell.shape[0], axis=0), 
                             W // gy_cell.shape[1], axis=1)[:H, :W]
        
        # Calculate dot product
        return (XX - ix) * gx_resized + (YY - iy) * gy_resized
    
    # Calculate noise for all four corners
    n00 = dot_product(0, 0)
    n10 = dot_product(1, 0)
    n01 = dot_product(0, 1)
    n11 = dot_product(1, 1)
    
    # Interpolate
    u, v = fade(XX), fade(YY)
    lerp = lambda a,b,t: a*(1-t)+b*t
    return lerp(lerp(n00, n10, u), lerp(n01, n11, u), v)

def fractal(shape, base_res, octaves=4, persistence=0.5):
    img = np.zeros(shape); amp=1; freq=1
    for _ in range(octaves):
        img += amp*perlin2d(shape, (int(base_res[0]*freq), int(base_res[1]*freq)))
        amp *= persistence; freq *= 2
    img = (img - img.min())/(img.max()-img.min())   # normalise 0-1
    return img

base_res = 8  # base resolution
octaves = 4   # number of octaves
persistence = 0.5  # persistence value

# Generate a single fractal noise image
noise_img = fractal((H,W), (base_res,base_res), octaves, persistence)
# EVOLVE-BLOCK-END

# Calculate and print the score
score = fitness(noise_img, target_img)
print(f"Score: {score:.4f}")
