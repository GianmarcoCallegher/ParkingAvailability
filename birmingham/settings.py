import datetime
import os

DIR_PATH = os.path.dirname(os.path.abspath(__file__))

FREQ = '30min'

TIME_INTERVAL = (datetime.time(9, 0, 0), datetime.time(16, 30, 0))

MIN_N_STALLS = 0

PREDICTION_STEPS = 2

TS_SHIFTS = 2
NEIGHBOURS = 4
KM_EPSILON = [250, 500, 1000, 2000]

WEEKLY_MEAN = None
WEEKLY_STD = None
