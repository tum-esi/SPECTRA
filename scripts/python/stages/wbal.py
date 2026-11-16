import numpy as np

def wbal(image: np.ndarray, r_gain_factor: int, b_gain_factor: int):
    if image.shape != (image.shape[0], image.shape[1], 3):
        raise ValueError("Input image must be a 3-channel RGB image.")
    # Widen before the product, as in gain.dgain: an in-place uint8 multiply
    # wraps and np.clip never sees the real value. Matches the circuit's
    # R' = min(R * r_gain, 255), B' = min(B * b_gain, 255).
    # uint16 images clip at 65535 instead, as in gain.dgain.
    top, dtype = (65535, np.uint16) if image.dtype == np.uint16 else (255, np.uint8)
    out = image.astype(np.int32)
    out[:, :, 0] *= int(r_gain_factor)
    out[:, :, 2] *= int(b_gain_factor)
    return np.clip(out, 0, top).astype(dtype)
