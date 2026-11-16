# Initial stage, where the images are generated
# We will have very simple examples (uniform, and uniform with the CFA applied)


import numpy as np
from PIL import Image
import os

W = 1920
H = 1080
DIMENSIONS = (H,W)

# CFA specific constant values
R_CFA = 1.0
G_CFA = 0.6
B_CFA = 0.3

STRIDE = 1000

STUCK_BLACK = 0
STUCK_WHITE = 255

def generate_LUT(stride, width=W, height=H) -> np.ndarray:
    lut = np.zeros((height, width), dtype=bool)
    lut.ravel()[::stride] = True
    # Keep a 2-pixel border free of defects so every dead pixel has all 8
    # neighbours even with Bayer offsets (+/-2), keeping the fixed divisor n=8
    # in the DPC and a shift instead of a division in the circuit.
    # TODO: support border defects by padding the image before DPC.
    lut[:2, :] = lut[-2:, :] = False
    lut[:, :2] = lut[:, -2:] = False
    return lut

def apply_cfa(img: np.ndarray) -> np.ndarray:
    """Apply RGGB Bayer filter gains to a single-channel image."""
    out = img.astype(np.float32)
    out[0::2, 0::2] *= R_CFA  # even rows, even cols: R
    out[0::2, 1::2] *= G_CFA  # even rows, odd cols:  G
    out[1::2, 0::2] *= G_CFA  # odd rows,  even cols: G
    out[1::2, 1::2] *= B_CFA  # odd rows,  odd cols:  B
    return out.astype(np.uint8)




def generate_image_uniform(value: int, cfa: bool, width=W, height=H) -> np.ndarray:
    img = np.full((height, width), value, dtype=np.uint8)
    if cfa:
        img = apply_cfa(img)
    return img


def apply_defect(LUT: np.ndarray, clean_image: np.ndarray) -> np.ndarray:
    # Wherever the LUT marks a dead pixel, stamp the defect into the image
    defective = clean_image.copy()
    defective[LUT] = STUCK_BLACK
    return defective
    
   

