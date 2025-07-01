import copy
import math

from src.gesture import DataGestures
from src.datasample import DataSample
from src.tools import rand_fix_interval

ROT_ANGLE = math.pi / 4


def rand_gesture() -> DataGestures:
    tmp: DataGestures = DataGestures()
    return tmp.setAllPointsToRandom()


def zero_gesture() -> DataGestures:
    tmp: DataGestures = DataGestures()
    return tmp.setAllPointsToZero()


def make_new_sample_variation(sample: DataSample) -> DataSample:
    tmp: DataSample = copy.deepcopy(sample)

    # Create a variation
    return tmp.rotate_sample(
        ROT_ANGLE - (ROT_ANGLE / 2) + (rand_fix_interval(ROT_ANGLE / 2)),
        ROT_ANGLE - (ROT_ANGLE / 2) + (rand_fix_interval(ROT_ANGLE / 2)),
        rand_fix_interval(math.pi / 10)
    ).scale_sample(
        1 + rand_fix_interval(0.2),
        1 + rand_fix_interval(0.2),
        1 + rand_fix_interval(0.2)
    ).translate_sample(
        rand_fix_interval(0.01),
        rand_fix_interval(0.01),
        rand_fix_interval(0.01))
