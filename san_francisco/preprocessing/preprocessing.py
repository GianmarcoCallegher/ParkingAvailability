import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
from utils import compute_parks_distance

from .settings import *


def import_parks(df_parks_info):
    """
    Returns:
        pandas.DataFrame: Parks info dataframe
    """
    df_parks = pd.read_csv(PARKS_RAW_DATA_PATH, sep=';')

    df_parks.columns = PARKS_COLUMNS

    df_parks['ParkAddress'] = df_parks['ParkAddress'].astype(str)

    df_parks['Longitude'] = df_parks[['StartX', 'EndX']].mean(axis=1)

    df_parks['Latitude'] = df_parks[['StartY', 'EndY']].mean(axis=1)

    df_parks = df_parks.drop(columns=['StartX', 'StartY','EndX','EndY'])

    df_parks = pd.merge(df_parks, df_parks_info, on=['ParkAddress'])

    return df_parks.sort_values(by='ParkAddress').reset_index(drop=True)


def merge_files():
    df_stopovers = pd.DataFrame()

    for data in STOPOVERS_RAW_DATA_PATH:
        df = pd.read_csv(data, sep=';')

        df = df[STOPOVERS_DEFAULT_COLUMNS]

        df.columns = STOPOVERS_COLUMNS
        
        df_stopovers = df_stopovers.append(df)

    return df_stopovers.reset_index(drop=True)


def import_stopovers():
    """
    Reads the stopovers dataset whose path is contained in stopovers_raw_data_path (see settings.py), 
    removes nulls , removes the observations outside the data range [from_date, to_date) 
    (see settings.py).

    Returns:
        (pandas.DataFrame, pandas.DataFrame): Stopovers dataframe, Partial parks's info dataframe
    """

    # df_stopovers = pd.read_csv(STOPOVERS_RAW_DATA_PATH, sep=';')

    # df_stopovers = df_stopovers[STOPOVERS_DEFAULT_COLUMNS]

    # df_stopovers.columns = STOPOVERS_COLUMNS
    df_stopovers = merge_files()

    df_stopovers['ParkAddress'] = df_stopovers['ParkAddress'].astype(str)

    df_stopovers = df_stopovers.dropna()

    df_stopovers['DateHour'] = pd.to_datetime(df_stopovers['DateHour'], format='%Y-%m-%d %H:%M:%S')

    df_stopovers = df_stopovers[(df_stopovers['DateHour'].dt.date >=
                                 FROM_DATE) & (df_stopovers['DateHour'].dt.date < TO_DATE)]

    df_stopovers['Occupancy'] = df_stopovers['NumberOfStalls'].div(df_stopovers['OccupiedStalls'])
    df_stopovers['Occupancy'] = df_stopovers['Occupancy'].fillna(0)

    df_stopovers['OccupiedStalls'] = df_stopovers['OccupiedStalls'].round(0).astype(int)
    df_stopovers['NumberOfStalls'] = df_stopovers['NumberOfStalls'].round(0).astype(int)

    df_parks_info = df_stopovers[df_stopovers['NumberOfStalls'] == df_stopovers.groupby('ParkAddress')['NumberOfStalls'].transform('max')].drop_duplicates(
            subset=['ParkAddress', 'NumberOfStalls'], keep='first').reset_index(drop=True)[['ParkAddress','NumberOfStalls']]

    return df_stopovers, df_parks_info



def create_dataset(df_stopovers, df_parks, freq):
    """
    Divedes the interval [from_date, to_date) (see settings.py) in slots of fixed lenght determinde by freq. 
    For each park counts the number of arrivals and departures in each time slot and subtracts
    the latter from the former. Then a merge with parks's info dataframe is performed to obtain 
    the number of stalls for each park. A cumulative sum of the subtraction arrivals - departures
    gives the number of occupied stalls for each park in each time slot. The occupancy rate is 
    simply computed by dividing the number of occupied stalls by the capacity of the park.
    The function returns a dataframe with four fileds: ParkAddress, DateHour, OccupiedStalls, Occupancy.

    Args:
        df_stopovers (pandas.Dataframe): Stopovers dataframe
        df_parks (pandas.Dataframe): Parks info dataframe
        freq (str or DateOffset): frequency alias

    Returns:
        (pandas.Dataframe): Observations dataframe
    """
    X = pd.DataFrame(dtype=int)

    date_time_interval = pd.date_range(FROM_DATE, TO_DATE, freq=freq)[:-1]
    
    for park in df_parks['ParkAddress']:
        df = df_stopovers[df_stopovers['ParkAddress'] == park]
        
        capacity = df.groupby(pd.Grouper(key='DateHour', freq=freq))['NumberOfStalls'].mean().fillna(0).reset_index().iloc[:, 1]
        
        occupancy = df.groupby(pd.Grouper(key='DateHour', freq=freq))['OccupiedStalls'].mean().fillna(0).reset_index().iloc[:, 1]

        X_temp = pd.DataFrame(dtype=int)

        X_temp['DateHour'] = date_time_interval
        X_temp['ParkAddress'] = park

        X_temp['NumberOfStalls'] = capacity
        X_temp['OccupiedStalls'] = occupancy
        X_temp['Occupancy'] = occupancy.div(capacity).fillna(0)

        X_temp['OccupiedStalls'] = X_temp['OccupiedStalls'].round(0).astype(int)
        X_temp['NumberOfStalls'] = X_temp['NumberOfStalls'].round(0).astype(int)

        X = X.append(X_temp)

    # X = pd.merge(X, df_parks[['ParkAddress', 'NumberOfStalls']], how='inner', on=[
    #              'ParkAddress'])

    # X['Occupancy'] = X['OccupiedStalls'].div(data['NumberOfStalls'], axis=0)

    return X.sort_values(by=['ParkAddress', 'DateHour']).reset_index(drop=True)


def get_df_stopovers_parks():
    """
    Tries to read the preprocessed stopovers dataframe and the preprocessed parks's info dataframe 
    from the paths stored in stopovers_preprocessed_data_path and parks_preprocessed_data_path (see settings.py).
    If the files do not exist, calls import_stopovers, passes the returned partial parks's info dataframe
    to import_parks, reads from missing_coords_dict (see settings.py) the coordinates of the parks for which 
    they are not avaiable and changes them in df_parks. Finally saves the preprocessed stopovers dataframe
    and the preprocessed parks's info dataframe in stopovers_preprocessed_data_path and 
    parks_preprocessed_data_path.

    Returns:
        (pandas.DataFrame, pandas.DataFrame): Stopovers dataframe, Parks's info dataframe
    """
    try:
        df_stopovers = pd.read_pickle(STOPOVERS_PREPROCESSED_DATA_PATH)
        df_parks = pd.read_pickle(PARKS_PREPROCESSED_DATA_PATH)
    except IOError:
        df_stopovers, df_parks_info = import_stopovers()
        df_parks = import_parks(df_parks_info)

        df_stopovers.to_pickle(STOPOVERS_PREPROCESSED_DATA_PATH)
        df_parks.to_pickle(PARKS_PREPROCESSED_DATA_PATH)

    return df_stopovers, df_parks


def get_df_stopovers():
    try:
        df_stopovers = pd.read_pickle(STOPOVERS_PREPROCESSED_DATA_PATH)
    except IOError:
        df_stopovers, _ = import_stopovers()
    
    return df_stopovers


def get_df_parks_info():
    return get_df_stopovers_parks()[1]


def get_dataset(freq='15min'):
    """
    Deletes all the stopovers with a duration smaller than duration and creates the observations dataset.

    Args:
        freq (str or DateOffset, optional): frequency alias. Defaults to '15min'.

    Returns:
        (pandas.Dataframe): Observations dataframe
    """
    try:
        X = pd.read_pickle(PREPROCESSED_DATA_PATH + '/' + freq + '.pkl')
    except IOError:
        df_stopovers, df_parks = get_df_stopovers_parks()

        X = create_dataset(df_stopovers, df_parks, freq=freq)

        X.to_pickle(PREPROCESSED_DATA_PATH + '/' + freq + '.pkl')

    return X


def get_parks_distance(metric='route'):
    try: 
        distance_matrix = np.load(DATA_PATH + '/W_' + metric + '.npy')
    except:
        coords = get_df_parks_info()[['Latitude', 'Longitude']].to_numpy()

        distance_matrix = compute_parks_distance(coords, 'San Francisco, CA, USA', metric)

        np.save(DATA_PATH + '/W_' + metric + '.npy', distance_matrix)

    return distance_matrix
