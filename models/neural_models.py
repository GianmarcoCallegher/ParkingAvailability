import numpy as np

import torch
import torch.nn.functional as F

from torch.nn.parameter import Parameter

from torch_geometric_temporal.nn import GConvGRU


class GatedRecurrentUnit(torch.nn.Module):
    def __init__(self, input_size, hidden_size, prediction_steps):
        super(GatedRecurrentUnit, self).__init__()

        self.hidden_size = hidden_size
        self.prediction_steps = prediction_steps

        self.Wz = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.Uz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_z = self.bias = Parameter(torch.Tensor(hidden_size))

        self.Wr = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.Ur = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_r = self.bias = Parameter(torch.Tensor(hidden_size))

        self.Wh = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.Uh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_h = self.bias = Parameter(torch.Tensor(hidden_size))

        self.linear = torch.nn.Linear(hidden_size, prediction_steps)


    def forward(self, x):
        h = torch.zeros(x.shape[0], x.shape[1], self.hidden_size)

        z = torch.sigmoid(self.Wz(x) + self.Uz(h) + self.b_z)
        r = torch.sigmoid(self.Wr(x) + self.Uz(h) + self.b_r)
        h_tilde = torch.tanh(self.Wh(x) + self.Uh(h * r) + self.b_h)
        h = z * h + (1 - z) * h_tilde

        h = F.relu(h)

        return self.linear(h)


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, filters, prediction_steps, K=3):
        super(RecurrentGCN, self).__init__()
        self.recurrent = GConvGRU(node_features, filters, K=K)
        self.linear = torch.nn.Linear(filters, prediction_steps)

    def forward(self, x, edge_index, edge_weight):
        h = torch.squeeze(self.recurrent(x, edge_index, edge_weight))
        h = F.relu(h)
        h = self.linear(h)

        return h



class EarlyStopping():
    def __init__(self, patience=3, delta=0.005):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta


    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.model = model

        elif score < self.best_score + self.delta:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.model = model
            self.counter = 0
