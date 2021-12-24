import datetime
import os

DATA_PATH = os.path.dirname(os.path.abspath(__file__)) + '/data'

RAW_DATA_PATH = DATA_PATH + '/raw'
PREPROCESSED_DATA_PATH = DATA_PATH + '/preprocessed'

DATA_RAW_DATA_PATH = RAW_DATA_PATH + '/dataset.csv'
COORDS_DATA_PATH = RAW_DATA_PATH + '/parks_coords.csv'

PARKS_PREPROCESSED_DATA_PATH = PREPROCESSED_DATA_PATH + '/parks_info.pkl'

DATA_COLUMNS = ['ParkAddress', 'NumberOfStalls', 'OccupiedStalls', 'DateHour']

AVOID_PARKS = ['NIA North', 'BHMBRTARC01', 'BHMNCPNHS01']

MIN_HOUR = datetime.time(8, 0, 0)
MAX_HOUR = datetime.time(16, 30, 0)

PREPROCESSED_DATASET_PATH = PREPROCESSED_DATA_PATH + '/dataset.pkl'