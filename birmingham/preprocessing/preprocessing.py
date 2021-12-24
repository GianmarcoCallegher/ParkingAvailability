import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
from .settings import *
from utils import compute_parks_distance


def fix_missing_values(data):
    parks = list(data['ParkAddress'].unique())

    max_date = data[['ParkAddress', 'DateHour']].groupby('ParkAddress').max().min().dt.date.values[0]
    min_date = data[['ParkAddress', 'DateHour']].groupby('ParkAddress').min().max().dt.date.values[0]

    data = data[(data['DateHour'].dt.date >= min_date) & (data['DateHour'].dt.date <= max_date)]

    datetime_range = pd.date_range(data['DateHour'].min(), data['DateHour'].max(), freq='30min')
    datetime_range = datetime_range[(datetime_range.time >= MIN_HOUR) & (datetime_range.time <= MAX_HOUR)]

    X = pd.DataFrame()
    
    for park in parks:
        df = data[data['ParkAddress'] == park].copy()
        df.index = pd.DatetimeIndex(df.DateHour)
        df = df.drop(columns='DateHour')
        df = df.reindex(datetime_range)
        df.index.name = 'DateHour'
        df['ParkAddress'] = df['ParkAddress'].fillna(method='pad')
        df = df.reset_index()
        X = X.append(df)

    X['Hour'] = X['DateHour'].dt.time
    X['DayOfWeek'] = X['DateHour'].dt.dayofweek

    X['WeeklyMeanOccupied'] = X.groupby(['ParkAddress', 'DayOfWeek', 'Hour'])['Occupancy'].transform(lambda x: x.rolling(8, min_periods=1).mean())

    X['WeeklyMeanOccupancy'] = X.groupby(['ParkAddress', 'DayOfWeek', 'Hour'])['Occupancy'].transform(lambda x: x.rolling(8, min_periods=1).mean())

    X['Occupancy'] = X['Occupancy'].fillna(X['WeeklyMeanOccupancy'])

    max_date = X[X['Occupancy'].isna()]['DateHour'].max()
    X = X[X['DateHour'] > max_date]

    X['OccupiedStalls'] = X['OccupiedStalls'].fillna(X['WeeklyMeanOccupied']).round().astype(int)

    X = X.drop(columns=['Hour', 'DayOfWeek', 'WeeklyMeanOccupied', 'WeeklyMeanOccupancy'])

    return X.reset_index(drop=True)


def create_dataset():
    data = pd.read_csv(DATA_RAW_DATA_PATH, sep=',', parse_dates=['LastUpdated'])

    data.columns = DATA_COLUMNS

    data = data.drop_duplicates()

    data['DateHour'] = data['DateHour'].dt.round('30T')

    data['Occupancy'] = data['OccupiedStalls'].div(data['NumberOfStalls'])

    parks = list(get_df_parks_info()['ParkAddress'])

    data = data[data['ParkAddress'].isin(parks)]

    data = data.drop(columns=['NumberOfStalls'])

    data.loc[data['DateHour'].dt.time < MIN_HOUR, 'DateHour'] = data.loc[data['DateHour'].dt.time < MIN_HOUR, 'DateHour'].apply(lambda x: x.replace(hour=MIN_HOUR.hour, minute=MIN_HOUR.minute))
    data.loc[data['DateHour'].dt.time > MAX_HOUR, 'DateHour'] = data.loc[data['DateHour'].dt.time > MAX_HOUR, 'DateHour'].apply(lambda x: x.replace(hour=MAX_HOUR.hour, minute=MAX_HOUR.minute))

    data = data.drop_duplicates(subset=['ParkAddress', 'DateHour'], keep='last')

    data = data.sort_values(by=['ParkAddress', 'DateHour'])

    data = fix_missing_values(data)

    data.loc[data['Occupancy'] < 0, 'Occupancy'] = 0
    data.loc[data['Occupancy'] > 1, 'Occupancy'] = 1

    return data.reset_index(drop=True)


def get_df_parks_info():
    try:
        df_parks = pd.read_pickle(PARKS_PREPROCESSED_DATA_PATH)
    except IOError:
        data = pd.read_csv(DATA_RAW_DATA_PATH, sep=',', parse_dates=['LastUpdated'])
        
        data.columns = DATA_COLUMNS

        df_parks = data[['ParkAddress', 'NumberOfStalls']].drop_duplicates()

        df_parks = df_parks[~df_parks['ParkAddress'].isin(AVOID_PARKS)].reset_index(drop=True)

        coords = pd.read_csv(COORDS_DATA_PATH, sep=',')

        df_parks = df_parks.merge(coords, on='ParkAddress', how='inner')

        df_parks.to_pickle(PARKS_PREPROCESSED_DATA_PATH)
    
    return df_parks


def get_parks_distance(metric='route'):
    try: 
        distance_matrix = np.load(DATA_PATH + '/W_' + metric + '.npy')
    except:
        coords = get_df_parks_info()[['Longitude', 'Latitude']].to_numpy()

        distance_matrix = compute_parks_distance(coords, 'Birmingham, UK', metric)

        np.save(DATA_PATH + '/W_' + metric + '.npy', distance_matrix)

    return distance_matrix


def get_dataset():
    try:
        X = pd.read_pickle(PREPROCESSED_DATASET_PATH)
    except IOError:
        X = create_dataset()

        X.to_pickle(PREPROCESSED_DATASET_PATH)

    return X

