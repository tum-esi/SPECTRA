# Static DPC with LUT.
# We assume that the manufacturer loads in the camera a static LUT, where it indicates which pixels are dead.
# if LUT = 1 -> the pixel is dead, else it is healty
# A pixel can be stuck black or stuck white, indipendently of this if there is a 1 it needs to be corrected
# for simplicity, we will assume a uniform stride of dead pixels, so that every STRIDE pixel will be dead 
# If we have a border, we will consider the only existing neighbours. 


import numpy as np

# TODO: This only works for grayscale images, not for bayer images 
NN_OFFSETS_2D = {(0,1), (1, 0), (1, 1), (0, -1), (-1, 0), (-1,-1), (-1,1), (1,-1)}
NN_OFFSETS_BAYER = {(0,2), (2, 0), (2, 2), (0, -2), (-2, 0), (-2,-2), (-2,2), (2,-2)}



def dpc(defective: np.ndarray, lut: np.ndarray, isBayer: bool = False) -> np.ndarray:
    corrected = defective.copy()
    offsets = NN_OFFSETS_BAYER if isBayer else NN_OFFSETS_2D
    # The LUT generator guarantees a defect-free border as wide as the offset
    # reach (2 for Bayer, 1 for grayscale) and no two adjacent defects, so
    # every dead pixel has exactly 8 healthy neighbours.
    # TODO: to allow border defects, pad the image with replicated pixels instead.
    margin = 2 if isBayer else 1
    assert not (lut[:margin, :].any() or lut[-margin:, :].any()
                or lut[:, :margin].any() or lut[:, -margin:].any())
    for r, c in zip(*np.nonzero(lut)):        # coords of every dead pixel
        acc = 0
        for dr, dc in offsets:
            acc += int(defective[r + dr, c + dc])
        # Euclidean division: acc = q*8 + rem with 0 <= rem < 8; the pixel gets q.
        # Matches the circuit's division witness (quotient/remainder decomposition).
        corrected[r, c] = acc // 8
    return corrected




