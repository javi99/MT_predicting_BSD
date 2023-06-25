import xgboost as xgb
import pandas as pd
import datetime as dt
import os 
import matplotlib.pyplot as plt

os.chdir(os.getcwd() + '/models')

from baseline_utils import baseline_model
from optuna_utils import Optuna

# get data
data_dir = '/Users/erikagutierrez/Documents/BSE/Term_3/Masters_Thesis/MT_predicting_BSD/processing/storage_final/'

data = pd.read_csv(data_dir + 'bicimad_dataframe.csv')

data.drop(['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'Unnamed: 0_x', 'Unnamed: 0_y'], axis = 1, inplace = True)

data['time'] = pd.to_datetime(data['time'])

# define train test breakouts

forecast_cutoff_start = (max(data['time']) + dt.timedelta(days = -7)) + dt.timedelta(hours = 1) # start date of the last week of data
forecast_cutoff_end = forecast_cutoff_start + dt.timedelta(days = 7) + dt.timedelta(hours = -1) # end date of the last week of data

baseline = baseline_model() # get baseline model class functions

datasets = baseline.create_modeling_datasets(data, [(forecast_cutoff_start, forecast_cutoff_end)], .4) # result is a list of lists [train, validation, test] for each desired forecast period (need to add validation soon)

# decide what features to use and standardize them
features = ['reservations_count', 
            'light', 
            'total_bases', 
            'free_bases', 
            'no_available', 
            'dock_bikes', 
            #'time', 
            'day', 
            'month', 
            'year', 
            'hour', 
            'weekday', 
            '81', 
            '82', 
            '83', 
            '86', 
            '87', 
            '88', 
            '89', 
            'work_day_indicator', 
            'covid_indicator']

target_plugs = 'plugs_count'
target_unplugs = 'unplugs_count'

feature_datasets = baseline.create_feature_datasets(datasets, features, 'standard') # create and standardize the train and test feature datasets  
plug_target_datasets = baseline.create_target_datasets(datasets, target_plugs) # create the train and test plug target vectors
unplug_target_datasets = baseline.create_target_datasets(datasets, target_unplugs) # create the train and test unplug target vectors

# train models

# XGBOOST

# Get DMatrices 

dmatrices_plugs = []
dmatrices_unplugs = []

for n in range(len(datasets)):
    dtrain_plugs = xgb.DMatrix(feature_datasets[n][0], label=plug_target_datasets[n][0])
    dvalid_plugs = xgb.DMatrix(feature_datasets[n][1], label=plug_target_datasets[n][1])
    dtest_plugs = xgb.DMatrix(feature_datasets[n][2], label=plug_target_datasets[n][2])
    dmatrices_plugs.append([dtrain_plugs, dvalid_plugs, dtest_plugs])
    
    dtrain_unplugs = xgb.DMatrix(feature_datasets[n][0], label=unplug_target_datasets[n][0])
    dvalid_unplugs = xgb.DMatrix(feature_datasets[n][1], label=unplug_target_datasets[n][1])
    dtest_unplugs = xgb.DMatrix(feature_datasets[n][2], label=unplug_target_datasets[n][2])
    dmatrices_unplugs.append([dtrain_unplugs, dvalid_unplugs, dtest_unplugs])


# train test and review

for n in range(len(datasets)):

    # hyperparameter tuning
    print('Starting hyperparameter search...')

    parameter_search_plugs = Optuna(dmatrices_plugs[n][0], dmatrices_plugs[n][1], dmatrices_plugs[n][2])
    parameter_search_unplugs = Optuna(dmatrices_unplugs[n][0], dmatrices_unplugs[n][1], dmatrices_unplugs[n][2])

    optimal_parameters_plugs = parameter_search_plugs.conduct_study('xgb')
    optimal_parameters_unplugs = parameter_search_unplugs.conduct_study('xgb')

    # train model with optimal parameters
    print('Training model with optimal parameters...')
    xgb_plugs = xgb.XGBRegressor(**optimal_parameters_plugs, objective='reg:squarederror').fit(feature_datasets[n][0], plug_target_datasets[n][0])
    xgb_unplugs = xgb.XGBRegressor(**optimal_parameters_unplugs, objective='reg:squarederror').fit(feature_datasets[n][0], unplug_target_datasets[n][0])

    # predict target
    print('Predicting...')
    plug_preds = xgb_plugs.predict(feature_datasets[n][2])
    unplug_preds = xgb_unplugs.predict(feature_datasets[n][2])

    # compute results
    print("Computing results...")
    results = []
    results.append(baseline.evaluate_metrics(plug_target_datasets[n][2], plug_preds, 'XGBOOST', 'Plugs'))
    results.append(baseline.evaluate_metrics(unplug_target_datasets[n][2], unplug_preds, 'XGBOOST', 'Unplugs'))

    pd.DataFrame(results, columns = ['Model Name', 'Target Feature','RSME', 'MAE', 'R2'])

# compute feature importances

baseline.display_feature_importance(xgb_plugs, feature_datasets[0][0])


















    