import pandas as pd
import numpy as np

from statsmodels.tsa.api import VAR

from .losses import *
from utils import *


class VectorAutoRegression():
    def __init__(self, data_loader, predictions_steps, order):
        self.data_loader = data_loader
        self.order = order
        self.model = None
        self.predictions_steps = predictions_steps


    def train(self, train_split):
        print('\nTrain\n')

        n = self.data_loader.data.shape[0]

        Y_train = self.data_loader.data.iloc[:int(n * train_split)].reset_index(drop=True)

        self.model = VAR(Y_train).fit(self.order)


    def test(self, test_split):
        print('\nTest\n')

        parks = list(self.data_loader.data.columns)

        n = self.data_loader.data.shape[0]

        Y_test = self.data_loader.data.iloc[-(self.order + int(n * test_split)):].reset_index(drop=True)

        test_dates = self.data_loader.data.index.values[-Y_test.shape[0]:]

        predictions = {park: {prediction_step: pd.DataFrame() for prediction_step in range(self.predictions_steps)} for park in parks}

        for i in range(self.order, Y_test.shape[0] - self.predictions_steps):
            y = Y_test.iloc[i:i + self.predictions_steps].to_numpy()
            y_hat = self.model.forecast(Y_test.iloc[i - self.order:i].to_numpy(), self.predictions_steps)

            for j in range(len(parks)):
                for prediction_step in range(self.predictions_steps):
                    predictions[parks[j]][prediction_step] = predictions[parks[j]][prediction_step].append({
                        'DateHour': test_dates[i + prediction_step], 
                        'Occupancy': y[prediction_step][j],
                        'Predicted Occupancy': y_hat[prediction_step][j]
                    }, ignore_index=True)
        

        avg_loss = np.zeros(self.predictions_steps)
        test_loss = {park: [] for park in parks}

        for park in parks:
            for prediction_step in range(self.predictions_steps):
                y = predictions[park][prediction_step]['Occupancy'].values
                y_hat = predictions[park][prediction_step]['Predicted Occupancy'].values

                l = float(root_mean_squared_error_loss(y, y_hat))

                test_loss[park].append(l)
                avg_loss[prediction_step] += l

        avg_loss /= len(parks)

        return list(avg_loss), test_loss, predictions

