#!/usr/bin/env python3

from enum import Enum, IntEnum, auto


class Keys(Enum):
    STATES = auto()
    OBS = auto()
    NEXT_STATES = auto()
    LOG_LIK = auto()
    LOG_PRIOR = auto()
    LOG_PROPOSAL = auto()
    IDX = auto()
    RESAMPLED = auto()
    DONE = auto()
    GT_TRAJECTORY = auto()
    STATE_DIM = auto()


class Columns(IntEnum):
    STATE_IDX = 0
    OBS_IDX = 1


class VizKeys(Enum):
    STATE_MEAN = auto()
    STATE_UPPER_STD = auto()
    STATE_LOWER_STD = auto()
    STATE_STD = auto()
    LOG_EVIDENCE = auto()
    TIME = auto()
