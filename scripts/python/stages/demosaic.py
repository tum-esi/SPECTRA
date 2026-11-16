import numpy as np


def NearestNeighbour(superpixel: np.ndarray, bayer_pattern: str) -> np.ndarray:
    """Expand a 2x2 Bayer superpixel into 2x2 RGB via nearest-neighbour fill.

    RGGB layout:
        R  G1
        G2 B

    R and B are shared by all four pixels; each row keeps its nearest green.
    """
    if bayer_pattern != "RGGB":
        raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")

    r, g1, g2, b = superpixel.ravel()

    rgb = np.empty((2, 2, 3), dtype=superpixel.dtype)
    rgb[..., 0] = r
    rgb[0, :, 1] = g1
    rgb[1, :, 1] = g2
    rgb[..., 2] = b
    return rgb


def demosaic(bayer_image: np.ndarray, bayer_pattern: str = "RGGB") -> np.ndarray:
    """Nearest-neighbour demosaic: apply NearestNeighbour to every 2x2 superpixel.

    Expects even height and width (whole superpixels only).
    """
    h, w = bayer_image.shape
    if h % 2 or w % 2:
        raise ValueError(f"Image dimensions must be even, got {h}x{w}")

    rgb = np.empty((h, w, 3), dtype=bayer_image.dtype)
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            rgb[y:y + 2, x:x + 2] = NearestNeighbour(
                bayer_image[y:y + 2, x:x + 2], bayer_pattern
            )
    return rgb

