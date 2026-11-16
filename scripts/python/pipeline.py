import argparse
import numpy as np
from stages import acquisition, deadpixel, gain, demosaic, wbal
from helpers import helpers
from enum import Enum




STAGES = Enum('Stage', [('DPC', 1),('GAIN', 2),('DEMOSAIC', 3),('WBAL', 4),])
ENABELED_STAGES = [1,2,3,4]

IS_BAYER = True


def even_dimension(value):
    value = int(value)
    if value < 4 or value % 2:
        raise argparse.ArgumentTypeError("dimensions must be even integers >= 4")
    return value

def main(width, height):
    print("-- Stage 0: Acquisition --")
    image = acquisition.generate_image_uniform(100 , IS_BAYER, width, height)
    lut = acquisition.generate_LUT(1000, width, height)
    defect_image = acquisition.apply_defect(lut,image)

    helpers.save_image("lut", lut.astype(np.uint8), stage="0_acquisition")
    helpers.save_image("uniform_cfa_no_defect", image, stage="0_acquisition")
    helpers.save_image("uniform_cfa_defect", defect_image, stage="0_acquisition")

    print("Macro Stage 1: On Chip Processing")

    print("\t -- Stage 1: Dead Pixel Correction")

    corrected_image =  deadpixel.dpc(defective=defect_image, lut=lut, isBayer=IS_BAYER)
    helpers.save_image("dpc_cfa", corrected_image, stage="1_dpc")

    print("\t -- Stage 2: Gain and clip --")
    
    gained_image = gain.dgain(corrected_image, 2)
    helpers.save_image("gained_image", gained_image, stage="2_gain")
    
    print("\t -- Stage 3: Demosaic --")

    demosaic_image = demosaic.demosaic(gained_image)
    helpers.save_image("demosaic_rgb", demosaic_image, stage="3_demosaic")
    
    print("\t -- Stage 4: White Balance --")
    white_balanced_image = wbal.wbal(demosaic_image, 2, 4)
    helpers.save_image("white_balanced_image", white_balanced_image, stage="4_wbal")
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Supernova fixture data")
    parser.add_argument("--width", type=even_dimension, default=acquisition.W)
    parser.add_argument("--height", type=even_dimension, default=acquisition.H)
    args = parser.parse_args()
    main(args.width, args.height)
