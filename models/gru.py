import numpy as np
import pandas as pd

import torch
import torch.utils.data as Data
from torch.autograd import Variable

from .losses import *
from utils import *

from .neural_models import GatedRecurrentUnit, EarlyStopping


class GRU():
    def __init__(self, data_loader, prediction_steps, model_params):
        self.prediction_steps = prediction_steps
        self.data_loader = data_loader
        self.model_params = model_params
        self.model = None


    def train(self, train_split, validation_split):
        X, Y, _ = self.data_loader.create_np_dataset()

        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)

        n = X.shape[0]

        train_index = int(n * train_split)
        validation_index = int(n * (train_split + validation_split))

        X_train = X[:train_index, :, :]
        Y_train = Y[:train_index, :]

        X_validation = X[train_index:validation_index, :, :]
        Y_validation = Y[train_index:validation_index, :]

        batch_size = self.model_params['batch_size']

        train_loader = Data.DataLoader(
            dataset=Data.TensorDataset(X_train, Y_train), 
            batch_size=batch_size
        )

        validation_loader = Data.DataLoader(
            dataset=Data.TensorDataset(X_validation, Y_validation), 
            batch_size=batch_size
        )

        model = GatedRecurrentUnit(input_size=X.shape[2], hidden_size=self.model_params['filters'], prediction_steps=self.prediction_steps)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.model_params['lr'])
        loss = torch.nn.MSELoss()

        early_stopping = EarlyStopping(patience=self.model_params['early_stopping_patience'], delta=self.model_params['early_stopping_delta'])

        print('\nTrain\n')
        
        for epoch in range(self.model_params['epochs']):
            train_loss = 0
            c = 0

            for _, (batch_x, batch_y) in enumerate(train_loader):
                y_hat = model(Variable(batch_x))
                l = torch.sqrt(loss(y_hat, Variable(batch_y)))

                train_loss += float(l) * batch_x.shape[0]
                c += batch_x.shape[0]

                l.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss /= c

            val_loss = 0
            c = 0

            for step, (batch_x, batch_y) in enumerate(validation_loader):
                y_hat = model(Variable(batch_x))
                l = torch.sqrt(loss(y_hat, Variable(batch_y)))

                val_loss += float(l) * batch_x.shape[0]
                c += batch_x.shape[0]

            val_loss /= c
            
            print('Epoch ' + str(epoch) + '/' + str(self.model_params['epochs']) + ': Train loss: ' + str(float(train_loss)) + ' Validation loss: ' + str(val_loss))

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                break

        
        self.model = early_stopping.model


    def test(self, test_split):
        parks = list(self.data_loader.data['ParkAddress'].unique())

        X, Y, dates = self.data_loader.create_np_dataset()

        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)

        n = X.shape[0]

        test_index = int(n * (1 - test_split))

        X_test = X[test_index:, :, :]
        Y_test = Y[test_index:, :]

        predictions = {park: {prediction_step: pd.DataFrame() for prediction_step in range(self.prediction_steps)} for park in parks}
        test_dates = dates[test_index:]

        print('\nTest\n')

        Y_hat_test = self.model(X_test)

        for prediction_step in range(self.prediction_steps):
            for j in range(len(parks)):
                for prediction_step in range(self.prediction_steps):
                    predictions[parks[j]][prediction_step] = pd.DataFrame({
                        'DateHour': test_dates, 
                        'Occupancy': Y_test[:, j, prediction_step].numpy(), 
                        'Predicted Occupancy': Y_hat_test[:, j, prediction_step].detach().numpy()
                        })


        avg_loss = np.zeros(self.prediction_steps)
        test_loss = {park: [] for park in parks}

        for park in parks:
            for prediction_step in range(self.prediction_steps):
                y = torch.tensor(predictions[park][prediction_step]['Occupancy']).numpy()
                y_hat = torch.tensor(predictions[park][prediction_step]['Predicted Occupancy']).numpy()

                l = float(root_mean_squared_error_loss(y, y_hat))

                test_loss[park].append(l)
                avg_loss[prediction_step] += l

        avg_loss /= len(parks)

        return list(avg_loss), test_loss, predictions


