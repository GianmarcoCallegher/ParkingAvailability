import json
import datetime
import os

DATA_PATH = os.path.dirname(os.path.abspath(__file__)) + '/data'
RAW_DATA_PATH = DATA_PATH + '/raw'
PREPROCESSED_DATA_PATH = DATA_PATH + '/preprocessed'

STOPOVERS_DEFAULT_COLUMNS = ['timestamp', 'segmentid', 'capacity', 'occupied']

STOPOVERS_COLUMNS = ['DateHour', 'ParkAddress', 'NumberOfStalls', 'OccupiedStalls']

PARKS_COLUMNS = ['ParkAddress', 'Street', 'StartX', 'StartY', 'EndX', 'EndY']

STOPOVERS_RAW_DATA_PATH = [
    RAW_DATA_PATH + '/sfpark_filtered_136_247_100taxis.csv',
    RAW_DATA_PATH + '/sfpark_filtered_136_247_200taxis.csv',
    RAW_DATA_PATH + '/sfpark_filtered_136_247_300taxis.csv',
    RAW_DATA_PATH + '/sfpark_filtered_136_247_400taxis.csv',
    RAW_DATA_PATH + '/sfpark_filtered_136_247_486taxis.csv'
]

PARKS_RAW_DATA_PATH = RAW_DATA_PATH + '/sfpark_filtered_segments.csv'

STOPOVERS_PREPROCESSED_DATA_PATH = PREPROCESSED_DATA_PATH + '/stopovers.pkl'
PARKS_PREPROCESSED_DATA_PATH = PREPROCESSED_DATA_PATH + '/parks_info.pkl'

FROM_DATE = datetime.date(2013, 6, 13)
TO_DATE = datetime.date(2013, 7, 24)