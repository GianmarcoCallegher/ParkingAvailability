import numpy as np
import pandas as pd

from data_loader import *

from pygam import LinearGAM

from .losses import *
from utils import *


class GAM():
    def __init__(self, data_loader, predictions_steps):
        self.models = {}
        self.predictions_steps = predictions_steps
        self.data_loader = data_loader


    def train(self, train_split):
        parks = self.data_loader.data['ParkAddress'].unique()

        self.models = {park: {} for park in parks}

        for prediction_step in range(self.predictions_steps):
            if prediction_step > 0:
                train_data = self.data_loader.shift_features(-prediction_step).split_data(0.0, train_split).data
            else:
                train_data = self.data_loader.split_data(0.0, train_split).data

            for park in parks:
                print('\nTrain park ' + park + ' T+' + str(prediction_step) + '\n')
                
                train_data_park = train_data[train_data['ParkAddress'] == park].copy().reset_index(drop=True)
                # train_data_park = train_data_park.iloc[:-prediction_steps]
                # train_data_park = train_data_park.dropna().reset_index(drop=True)

                train_data_park = train_data_park.drop(columns=['ParkAddress', 'OccupiedStalls', 'DateHour'])

                x = train_data_park.drop(columns=['Occupancy'])
                y = train_data_park['Occupancy']

                model = LinearGAM().fit(x, y)

                self.models[park][prediction_step] = model


    def test(self, test_split):
        parks = self.data_loader.data['ParkAddress'].unique()

        test_loss = {park: [] for park in parks}
        predictions = {park: {} for park in parks}

        avg_loss = np.zeros(self.predictions_steps)

        for prediction_step in range(self.predictions_steps):
            if prediction_step > 0:
                test_data = self.data_loader.shift_features(-prediction_step).split_data(1 - test_split, 1).data
            else:
                test_data = self.data_loader.split_data(1 - test_split, 1).data


            for park in parks:
                print('\nTest park ' + park + ' T+' + str(prediction_step) + '\n')

                predictions[park][prediction_step] = pd.DataFrame()

                test_data_park = test_data[test_data['ParkAddress'] == park].copy().reset_index(drop=True)
                test_data_park = test_data_park.iloc[:-self.predictions_steps]
                # test_data_park = test_data_park.dropna().reset_index(drop=True)

                test_dates = test_data_park['DateHour']

                test_data_park = test_data_park.drop(columns=['ParkAddress', 'OccupiedStalls', 'DateHour'])

                x = test_data_park.drop(columns=['Occupancy'])
                y = test_data_park['Occupancy']

                y_pred = self.models[park][prediction_step].predict(x)

                test_loss[park].append(root_mean_squared_error_loss(y.values, y_pred))

                predictions[park][prediction_step]['DateHour'] = test_dates
                predictions[park][prediction_step]['Occupancy'] = y.values
                predictions[park][prediction_step]['Predicted Occupancy'] = y_pred

                avg_loss[prediction_step] += test_loss[park][-1]

                print('\nTest loss: ' + str(test_loss[park][-1]) + '\n')

        avg_loss /= len(parks)

        return list(avg_loss), test_loss, predictions

