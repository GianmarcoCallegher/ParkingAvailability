import numpy as np


def root_mean_squared_error_loss(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


def mean_absolute_error_loss(y_true, y_pred):
    return np.mean(np.absolute(y_true - y_pred))
