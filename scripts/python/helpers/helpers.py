import numpy as np
from PIL import Image
from pathlib import Path

# All pipeline outputs live under scripts/python/out/<stage>/
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "out"

def save_image(name: str, array: np.ndarray, stage: str = "misc") -> Path:
    out_dir = OUTPUT_ROOT / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    # save the numpy array
    np.save(out_dir / f"{name}.npy", array)
    # save the image as PNG
    Image.fromarray(array).save(out_dir / f"{name}.png")
    # save the raw image
    array.astype('uint8').tofile(out_dir / f"{name}_uint8.raw")
    return out_dir
