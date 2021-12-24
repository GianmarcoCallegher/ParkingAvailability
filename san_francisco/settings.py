import datetime
import os

DIR_PATH = os.path.dirname(os.path.abspath(__file__))

FREQ = '15min'

DATE_INTERVAL = (datetime.datetime(2016, 10, 4), datetime.datetime(2016, 12, 19))

PREDICTION_STEPS = 4

TS_SHIFTS = 4
NEIGHBOURS = 4
KM_EPSILON =  [50, 100, 250, 500]

WEEKLY_MEAN = 4
WEEKLY_STD = 4
