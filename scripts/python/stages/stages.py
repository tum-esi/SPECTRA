import acquisition,deadpixel,demosaic,gain,wbal
from enum import Enum

class Stages(Enum):
    DPC = 1,
    GAIN = 2,
    DEMOSAIC = 3,
    WBAL = 4
    