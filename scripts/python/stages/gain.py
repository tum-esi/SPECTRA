import numpy as np

def dgain(image: np.ndarray, gain_factor: int):
    # Widen before the product: a uint8 array times a Python int stays uint8,
    # so px*g wraps and np.clip never sees the real value. Matches the circuit,
    # which decomposes px*g into 16 bits and selects out = min(px*g, 255).
    # uint16 raws (real sensor data, GTSRB) clip at 65535 instead; the circuit is 8-bit only.
    top, dtype = (65535, np.uint16) if image.dtype == np.uint16 else (255, np.uint8)
    return np.clip(image.astype(np.int32) * int(gain_factor), 0, top).astype(dtype)
