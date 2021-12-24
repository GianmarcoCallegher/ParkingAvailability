import numpy as np
import pandas as pd

from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.signal import StaticGraphTemporalSignalBatch

import torch

from .losses import *
from utils import *

from .neural_models import RecurrentGCN, EarlyStopping




class GCNN():
    def __init__(self, data_loader, prediction_steps, W, model_params, epsilon):
        self.data_loader = data_loader
        self.prediction_steps = prediction_steps
        self.W = W
        self.model_params = model_params
        self.epsilon = epsilon
        self.model = None

        self.edge_index = None
        self.edge_weight = None


    @staticmethod
    def generate_batches(f, t, ei, ew, node_count, batch_size):
        features = []
        targets = []
        edge_indeces = []
        edge_weights = []
        batches = []

        for i in range(0, len(f) - batch_size, batch_size):
            features_s = []
            targets_s = []

            for j in range(batch_size):
                features_s.append(f[i + j])
                targets_s.append(t[i + j])
            
            features.append(np.concatenate(features_s))
            targets.append(np.concatenate(targets_s))

        for i in range(batch_size):
            edge_indeces.append(ei + node_count * i)
            edge_weights.append(ew)
            batches.append(np.array([i for _ in range(node_count)]))

        edge_indeces = np.concatenate(edge_indeces, axis=1)
        edge_weights = np.concatenate(edge_weights)
        batches = np.concatenate(batches)

        return StaticGraphTemporalSignalBatch(edge_index=edge_indeces, edge_weight=edge_weights, features=features, targets=targets, batches=batches)
    

    @staticmethod
    def generate_edge_indeces(W, epsilon):
        edge_index = [[],[]]
        edge_weight = []

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                if i != j and W[i][j] < epsilon:
                    edge_index[0].append(i)
                    edge_index[1].append(j)
                    edge_weight.append(np.exp(-W[i][j] / 1000))
        
        return np.array(edge_index), np.array(edge_weight)


    def train(self, train_split, validation_split):
        X, Y, _ = self.data_loader.create_np_dataset()

        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)

        self.edge_index, self.edge_weight = self.generate_edge_indeces(self.W, self.epsilon)

        batch_size = self.model_params['batch_size']

        dataset = self.generate_batches(X, Y, self.edge_index, self.edge_weight, self.W.shape[0], batch_size)

        train_dataset, validation_dataset = temporal_signal_split(dataset, train_ratio=train_split)
        validation_dataset, _ = temporal_signal_split(validation_dataset, train_ratio=validation_split / (1 - train_split))

        model = RecurrentGCN(node_features=X.shape[2], filters=self.model_params['filters'], prediction_steps=self.prediction_steps)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.model_params['lr'])
        loss = torch.nn.MSELoss()

        early_stopping = EarlyStopping(patience=self.model_params['early_stopping_patience'], delta=self.model_params['early_stopping_delta'])

        print('\nTrain\n')

        for epoch in range(self.model_params['epochs']):
            train_loss = 0
            c = 0

            for _, snapshot in enumerate(train_dataset):
                y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
                l = torch.sqrt(loss(y_hat, snapshot.y))

                train_loss += float(l)
                c += 1

                l.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss /= c

            val_loss = 0
            c = 0

            for _, snapshot in enumerate(validation_dataset):
                y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
                l = torch.sqrt(loss(y_hat, snapshot.y))

                val_loss += float(l)
                c += 1

            val_loss /= c
            
            print('Epoch ' + str(epoch) + '/' + str(self.model_params['epochs']) + ': Train loss: ' + str(float(train_loss)) + ' Validation loss: ' + str(val_loss))

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                break

        
        self.model = early_stopping.model


    def test(self, test_split):
        parks = list(self.data_loader.data['ParkAddress'].unique())

        X, Y, dates = self.data_loader.create_np_dataset()

        batch_size = self.model_params['batch_size']

        dataset = self.generate_batches(X, Y, self.edge_index, self.edge_weight, self.W.shape[0], batch_size)

        tv_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=(1 - test_split))

        predictions = {park: {prediction_step: pd.DataFrame() for prediction_step in range(self.prediction_steps)} for park in parks}

        test_dates = dates[len(tv_dataset.features) * batch_size:len(dataset.features) * batch_size]
        # test_dates = dates[-len(test_dataset.features):]

        timedelta = self.data_loader.data.iloc[1]['DateHour'] - self.data_loader.data.iloc[0]['DateHour']
        date_index = 0

        print('\nTest\n')

        for _, snapshot in enumerate(test_dataset):
            y_hat = self.model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        
            for i in range(0, snapshot.x.shape[0], len(parks)):
                for j in range(len(parks)):
                    for prediction_step in range(self.prediction_steps):

                        predictions[parks[j]][prediction_step] = predictions[parks[j]][prediction_step].append({
                            'DateHour': test_dates[date_index] + prediction_step * timedelta, 
                            'Occupancy': float(snapshot.y[i + j][prediction_step]),
                            'Predicted Occupancy': float(y_hat[i + j][prediction_step])
                        }, ignore_index=True)

                date_index += 1

        avg_loss = np.zeros(self.prediction_steps)
        test_loss = {park: [] for park in parks}

        for park in parks:
            for prediction_step in range(self.prediction_steps):
                y = predictions[park][prediction_step]['Occupancy'].values
                y_hat = predictions[park][prediction_step]['Predicted Occupancy'].values

                l = float(root_mean_squared_error_loss(y, y_hat))

                test_loss[park].append(l)
                avg_loss[prediction_step] += l

        avg_loss /= len(parks)

        return list(avg_loss), test_loss, predictions
        