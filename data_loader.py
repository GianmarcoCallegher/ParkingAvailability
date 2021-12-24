import pandas as pd
import numpy as np

import copy

from utils import *


class UnivariateDataLoader():
    def __init__(self, data, df_parks_info=None, W=None):
        self.data = data.copy()
        self.df_parks_info = df_parks_info
        self.W = W

        self.rolling_weekly_mean = False
        self.rolling_weekly_std = False
        self.hour = False
        self.day_of_week = False
        self.month = False
        self.neighbours = None


    def create_extended_features(self, ts_shifts=1, mean=None, std=None, hour=False, day_of_week=False, month=False, neighbours=0):
        for i in range(1, ts_shifts + 1):
            self.data['Occupancy T-' + str(i)] = create_time_shift_features(self.data, i)

        dates_hours = self.data['DateHour']

        if mean is not None:
            self.data['RollingWeeklyMean'] = self.data.groupby(['ParkAddress', dates_hours.dt.day_of_week, dates_hours.dt.hour * 60 + dates_hours.dt.minute])['Occupancy'].transform(
                lambda x: x.shift(1).rolling(4).mean())

            self.rolling_weekly_mean = True

        if std:
            self.data['RollingWeeklyStd'] = self.data.groupby(['ParkAddress', dates_hours.dt.day_of_week, dates_hours.dt.hour * 60 + dates_hours.dt.minute])['Occupancy'].transform(
                lambda x: x.shift(1).rolling(4).std())
            self.rolling_weekly_std = True

        if hour:
            self.data['Hour'] = dates_hours.dt.hour * 60 + dates_hours.dt.minute
            self.hour = True
        
        if day_of_week:
            self.data['DayOfWeek'] = dates_hours.dt.day_of_week
            self.day_of_week = True

        if month:
            self.data['Month'] = dates_hours.dt.month
            self.month = True

        if neighbours > 0:
            self.neighbours = neighbours
            self.data = create_spatial_features(self.data, self.W, self.df_parks_info, self.neighbours)



    def copy_new_data(self, data):
        data_loader = copy.deepcopy(self)
        data_loader.data = data

        return data_loader
    

    def split_data(self, start, end):
        splitted_data = pd.DataFrame()

        for park in self.df_parks_info['ParkAddress']:
            # dropna().reset_index(drop=True)
            df = self.data[self.data['ParkAddress'] == park].dropna().reset_index(drop=True).copy()

            n = df.shape[0]

            start_idx = int(n * start)
            end_idx = int(n * end)

            splitted_data = splitted_data.append(df.iloc[start_idx:end_idx])

        splitted_data_loader = self.copy_new_data(splitted_data.reset_index(drop=True))

        return splitted_data_loader


    def shift_features(self, shift):
        shifted_data = self.data.copy()

        shifted_data['Occupancy'] = create_time_shift_features(shifted_data, shift, 'Occupancy')

        timedelta = shifted_data.iloc[1]['DateHour'] - shifted_data.iloc[0]['DateHour']

        shifted_data['DateHour'] = shifted_data['DateHour'] - (timedelta * shift)

        dates_hours = shifted_data['DateHour']

        if self.rolling_weekly_mean:
            shifted_data['RollingWeeklyMean'] = shifted_data.groupby(['ParkAddress', dates_hours.dt.day_of_week, dates_hours.dt.hour * 60 + dates_hours.dt.minute])['Occupancy'].transform(
                lambda x: x.shift(1).rolling(4).mean())

        if self.rolling_weekly_std:
            shifted_data['RollingWeeklyStd'] = shifted_data.groupby(['ParkAddress', dates_hours.dt.day_of_week, dates_hours.dt.hour * 60 + dates_hours.dt.minute])['Occupancy'].transform(
                lambda x: x.shift(1).rolling(4).std())


        if self.hour:
            shifted_data['Hour'] = dates_hours.dt.hour * 60 + dates_hours.dt.minute
        
        if self.day_of_week:
            shifted_data['DayOfWeek'] = dates_hours.dt.day_of_week

        if self.month:
            shifted_data['Month'] = dates_hours.dt.month

        return self.copy_new_data(shifted_data)


class VARDataLoader():
    def __init__(self, data):
        self.data = self.create_time_series(data.copy())

    
    def create_time_series(self, data):
        parks = list(data['ParkAddress'].unique())

        Y = pd.DataFrame(index=data['DateHour'].unique())

        for park in parks:
            Y[park] = data[data['ParkAddress'] == park]['Occupancy'].values


        return Y


class MultivariateDLDataLoader():
    def __init__(self, data, predictions_steps=None):
        self.data = data.copy()
        self.predictions_steps = predictions_steps if predictions_steps is not None else 0

        self.features_labels = []
        self.target_labels = []

        self.rolling_weekly_mean = False
        self.rolling_weekly_std = False
        self.hour = False
        self.day_of_week = False
        self.month = False
        self.neighbours = None


    def create_extended_features(self, ts_shifts=1, mean=None, std=None, hour=False, day_of_week=False, month=False):
        for i in range(1, ts_shifts + 1):
            self.data['Occupancy T-' + str(i)] = create_time_shift_features(self.data, i)
            self.features_labels.append('Occupancy T-' + str(i))

        if self.predictions_steps:
            for i in range(self.predictions_steps):
                self.data['Occupancy T+' + str(i)] = create_time_shift_features(self.data, -i)
                self.target_labels.append('Occupancy T+' + str(i))

        dates_hours = self.data['DateHour']

        if mean is not None:
            self.data['RollingWeeklyMean'] = self.data.groupby(['ParkAddress', dates_hours.dt.day_of_week, dates_hours.dt.hour * 60 + dates_hours.dt.minute])['Occupancy'].transform(
                lambda x: x.shift(1).rolling(4).mean())

            for i in range(self.predictions_steps):
                self.data['RollingWeeklyMean T+' + str(i)] = create_time_shift_features(self.data, -i, 'RollingWeeklyMean')
                self.features_labels.append('RollingWeeklyMean T+' + str(i))

            self.data = self.data.drop(columns='RollingWeeklyMean')

            self.rolling_weekly_mean = True

        if std:
            self.data['RollingWeeklyStd'] = self.data.groupby(['ParkAddress', dates_hours.dt.day_of_week, dates_hours.dt.hour * 60 + dates_hours.dt.minute])['Occupancy'].transform(
                lambda x: x.shift(1).rolling(4).std())

            for i in range(self.predictions_steps):
                self.data['RollingWeeklyStd T+' + str(i)] = create_time_shift_features(self.data, -i, 'RollingWeeklyStd')
                self.features_labels.append('RollingWeeklyStd T+' + str(i))

            self.data = self.data.drop(columns='RollingWeeklyStd')

            self.rolling_weekly_std = True

        if hour:
            self.data['Hour'] = dates_hours.dt.hour * 60 + dates_hours.dt.minute
            self.features_labels.append('Hour')
            self.hour = True
        
        if day_of_week:
            self.data['DayOfWeek'] = dates_hours.dt.day_of_week
            self.features_labels.append('DayOfWeek')
            self.day_of_week = True

        if month:
            self.data['Month'] = dates_hours.dt.month
            self.features_labels.append('Month')
            self.month = True

    @staticmethod
    def normalize_features(X):
        for i in range(X.shape[2]):
            if X[:, :, i].max() > 1:
                # X[:, :, i] = (X[:, :, i] - X[:, :, i].min()) / (X[:, :, i].max() - X[:, :, i].min())

                X[:, :, i] /= X[:, :, i].max()

        return X


    def create_np_dataset(self, normalize=True):
        parks = list(self.data['ParkAddress'].unique())

        df = self.data.dropna().copy()

        X = np.empty((len(df['DateHour'].unique()), len(parks), len(self.features_labels)))
        Y = np.empty((len(df['DateHour'].unique()), len(parks), len(self.target_labels)))
        
        for i in range(len(parks)):
            df1 = df[df['ParkAddress'] == parks[i]]

            X[:, i, :] =  df1[self.features_labels].to_numpy()
            Y[:, i, :] = df1[self.target_labels].values

        dates = df['DateHour'].unique()

        if normalize:
            X = self.normalize_features(X)

        return X, Y, dates


