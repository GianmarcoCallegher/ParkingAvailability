from models.lgbm import LGBM
from models.gam import GAM
from models.gcnn import GCNN
from models.gru import GRU
from models.var import VectorAutoRegression

from utils import *


from birmingham.preprocessing.preprocessing import *
from birmingham.settings import *

# from san_francisco.preprocessing.preprocessing import *
# from san_francisco.settings import *

from data_loader import *


lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'bagging_freq': 0,
    'learning_rate': 0.01,
    'num_leaves': 2**6,
    'n_estimators': 1000,
    'early_stopping_rounds': 100,
    'verbose': -1
}


net_params = {
    'lr': 0.001,
    'epochs': 1000,
    'early_stopping_patience': 100,
    'batch_size': 32,
    'filters': 16,
    'early_stopping_delta': 0.001
}


def gam(data, ts_shifts, neighbours, df_parks_info=None, W=None):
    data_loader = UnivariateDataLoader(data, df_parks_info, W)

    data_loader.create_extended_features(ts_shifts=ts_shifts, mean=WEEKLY_MEAN, std=WEEKLY_STD, hour=True, day_of_week=True, month=True, neighbours=neighbours)

    gam = GAM(data_loader, PREDICTION_STEPS)
    
    gam.train(train_split=0.6)


def lgbm(data, ts_shifts, neighbours, df_parks_info=None, W=None):
    data_loader = UnivariateDataLoader(data, df_parks_info, W)

    data_loader.create_extended_features(ts_shifts=ts_shifts, mean=WEEKLY_MEAN, std=WEEKLY_STD, hour=True, day_of_week=True, month=True, neighbours=neighbours)

    lgbm = LGBM(data_loader, PREDICTION_STEPS, lgb_params)
    
    feature_importance = lgbm.train(train_split=0.6, validation_split=0.2)


def var(data, order):
    data_loader = VARDataLoader(data)

    var = VectorAutoRegression(data_loader, PREDICTION_STEPS, order)
    
    var.train(train_split=0.6)


def gru(data, ts_shifts):
    data_loader = MultivariateDLDataLoader(data, PREDICTION_STEPS)

    data_loader.create_extended_features(ts_shifts=ts_shifts, mean=WEEKLY_MEAN, std=WEEKLY_STD, hour=True, day_of_week=True, month=True)

    gru = GRU(data_loader, PREDICTION_STEPS, net_params)

    gru.train(train_split=0.6, validation_split=0.2)

    avg_loss, test_loss, predictions = gru.test(test_split=0.2)



def gcnn(data, W, ts_shifts, km_epsilon):
    data_loader = MultivariateDLDataLoader(data, PREDICTION_STEPS)

    data_loader.create_extended_features(ts_shifts=ts_shifts, mean=WEEKLY_MEAN, std=WEEKLY_STD, hour=True, day_of_week=True, month=True)

    gcnn = GCNN(data_loader, PREDICTION_STEPS, W, net_params, km_epsilon)

    gcnn.train(train_split=0.6, validation_split=0.2)


def main():
    data = get_dataset()
    
    df_parks_info = get_df_parks_info()

    W = get_parks_distance()


    for ts_shift in range(1, TS_SHIFTS + 1):

        var(data, ts_shift)
        gru(data, ts_shift)

        for neighbour in range(NEIGHBOURS + 1):
            gam(data, ts_shift, neighbour, df_parks_info, W)
            lgbm(data, ts_shift, neighbour, df_parks_info, W)


        for km_epsilon in KM_EPSILON:
            gcnn(data, W, ts_shift, km_epsilon)    


main()


