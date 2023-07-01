import xgboost as xgb
import pandas as pd
import datetime as dt
import os 
import matplotlib.pyplot as plt
import time
import pickle

os.chdir(os.getcwd() + '/models')

from baseline_utils import baseline_model
from optuna_utils import Optuna



baseline = baseline_model() # get baseline model class functions

# Step 0: get data

data_dir = '/Users/erikagutierrez/Documents/BSE/Term_3/Masters_Thesis/MT_predicting_BSD/processing/storage_final/'

data = pd.read_csv(data_dir + 'XGBoost_data_not_standardized_or_splited.csv')

data.drop(['Unnamed: 0'], axis = 1, inplace = True)

data['time'] = pd.to_datetime(data['time'])

data['week_of_year'] = data['time'].apply(lambda x: int(x.strftime('%U')))

data['weekday'] = data['time'].apply(lambda x: int(x.isoweekday()))

data[data['weekday'] == 7].loc[:] = 0 # change sunday to start of week for proper sorting

data['hour'] = data['time'].apply(lambda x: x.hour)

# Step 1: define train test breakouts

timeperiods_text = ['INNOVA_comparison', 'all_of_2022']

train_start = dt.datetime.strptime('2019-01-06 00:00:00', '%Y-%m-%d %H:%M:%S')

## INNOVA Comparison Time Period

innovaweek_start = dt.datetime.strptime('2021-06-20 00:00:00', '%Y-%m-%d %H:%M:%S')
innovaweek_end = dt.datetime.strptime('2021-07-03 23:00:00', '%Y-%m-%d %H:%M:%S')

## All of 2022

start_2022 = dt.datetime.strptime('2022-02-01 00:00:00', '%Y-%m-%d %H:%M:%S')
end_2022 = dt.datetime.strptime('2022-12-31 23:00:00', '%Y-%m-%d %H:%M:%S')

timeperiods = [(innovaweek_start, innovaweek_end), (start_2022, end_2022)]

fixed_datasets = baseline.create_modeling_datasets(data, timeperiods, train_start)
sequential_datasets = baseline.create_weekly_sequential_datasets(fixed_datasets) 

# Step 2: Decide what features and make training matrices and target vectors

features = [
            'activate', 
            #'name', 
            'reservations_count', 
            'total_bases',
            'free_bases', 
            #'number', 
            'no_available', 
            #'address', 
            'dock_bikes',
            #'id_station', 
            #'time', 
            'year', 
            #'plugs_count', 
            #'unplugs_count',
            'latitude', 
            'longitude', 
            '83', 
            '86', 
            '87', 
            '88', 
            '89', 
            #'dia_semana',
            'work_day_indicator', 
            'covid_indicator', 
            #'index', 
            #'year_week_index',
            'month_sin', 
            'month_cos', 
            'day_sin', 
            'day_cos', 
            'hour_sin', 
            'hour_cos',
            'weekday_sin', 
            'weekday_cos', 
            'week_of_year_sin', 
            'week_of_year_cos',
            'wind_cos', 
            'wind_sin', 
            'light0', 
            'light1', 
            'light2', 
            'light3',
            #'week_of_year', 
            'weekday' 
            #'hour'
            ]


fixed_datasets_features = baseline.create_feature_datasets(fixed_datasets, features, 'fixed', 'standard')
fixed_datasets_plugs = baseline.create_target_datasets(fixed_datasets, 'plugs_count', 'fixed')
fixed_datasets_unplugs = baseline.create_target_datasets(fixed_datasets, 'unplugs_count', 'fixed')

sequential_datasets_features = baseline.create_feature_datasets(sequential_datasets, features, 'sequential', 'standard')
sequential_datasets_plugs = baseline.create_target_datasets(sequential_datasets, 'plugs_count', 'sequential')
sequential_datasets_unplugs = baseline.create_target_datasets(sequential_datasets, 'unplugs_count', 'sequential')


# Step 3: Get Results

now = dt.datetime.strftime(dt.datetime.now(), format = "%d-%m-%Y_%H:%M:%S")

os.chdir(os.getcwd() + '/trainings')

os.mkdir(f'run_{now}')


results = []

## Basic Baseline 

basic_baseline = baseline.basic_baseline(data)

basic_baseline = basic_baseline.dropna()

results.append(baseline.evaluate_metrics(basic_baseline['unplugs_count'], basic_baseline['pred_unplugs'], 'Basic Baseline', 'unplugs', 'all timeperiods'))
results.append(baseline.evaluate_metrics(basic_baseline['plugs_count'], basic_baseline['pred_plugs'], 'Basic Baseline', 'plugs', 'all timeperiods'))

## XGBOOST

### transform data to DMatrices for faster training
fixed_datasets_DMatrices_plugs = baseline.create_DMatrices(fixed_datasets_features, fixed_datasets_plugs, 'fixed')
fixed_datasets_DMatrices_unplugs = baseline.create_DMatrices(fixed_datasets_features, fixed_datasets_unplugs, 'fixed')
sequential_datasets_DMatrices_plugs = baseline.create_DMatrices(sequential_datasets_features, sequential_datasets_plugs, 'sequential')
sequential_datasets_DMatrices_unplugs = baseline.create_DMatrices(sequential_datasets_features, sequential_datasets_unplugs, 'sequential')

### fixed datasets

for t in range(len(timeperiods)):

    # hyperparameter tunings
    print('Starting hyperparameter search...')

    parameter_search_plugs = Optuna(fixed_datasets_DMatrices_plugs[t][0], fixed_datasets_DMatrices_plugs[t][1])
    parameter_search_unplugs = Optuna(fixed_datasets_DMatrices_unplugs[t][0], fixed_datasets_DMatrices_unplugs[t][1])

    optimal_parameters_plugs = parameter_search_plugs.conduct_study('xgb')
    optimal_parameters_unplugs = parameter_search_unplugs.conduct_study('xgb')

    # train model with optimal parameters
    print('Training model with optimal parameters...')
    xgb_plugs = xgb.XGBRegressor(**optimal_parameters_plugs, objective='reg:squarederror').fit(fixed_datasets_features[t][0], fixed_datasets_plugs[t][0])
    xgb_unplugs = xgb.XGBRegressor(**optimal_parameters_plugs, objective='reg:squarederror').fit(fixed_datasets_features[t][0], fixed_datasets_unplugs[t][0])

    # predict target
    print('Predicting...')
    #test datasets
    plug_preds_test = xgb_plugs.predict(fixed_datasets_features[t][1])
    unplug_preds_test = xgb_unplugs.predict(fixed_datasets_features[t][1])

    #train datasets
    plug_preds_train = xgb_plugs.predict(fixed_datasets_features[t][0])
    unplug_preds_train = xgb_unplugs.predict(fixed_datasets_features[t][0])

    # compute results
    print("Computing results...")
    results.append(baseline.evaluate_metrics(fixed_datasets_plugs[t][1], plug_preds_test, 'XGBOOST', 'Plugs', timeperiods_text[t], 'test set'))
    results.append(baseline.evaluate_metrics(fixed_datasets_unplugs[t][1], unplug_preds_test, 'XGBOOST', 'Unplugs', timeperiods_text[t], 'test set'))
    
    results.append(baseline.evaluate_metrics(fixed_datasets_plugs[t][0], plug_preds_train, 'XGBOOST', 'Plugs', timeperiods_text[t], 'train set'))
    results.append(baseline.evaluate_metrics(fixed_datasets_unplugs[t][0], unplug_preds_train, 'XGBOOST', 'Unplugs', timeperiods_text[t], 'train set'))

    print("Saving models and predictions....")
    xgb_plugs_path = f'run_{now}/plugs_fixed_{timeperiods_text[t]}.pkl'
    xgb_unplugs_path = f'run_{now}/unplugs_fixed_{timeperiods_text[t]}.pkl'
    with open(xgb_plugs_path, 'wb') as file:
        pickle.dump(xgb_plugs, file)
    with open(xgb_unplugs_path, 'wb') as file:
        pickle.dump(xgb_unplugs, file)
    pd.DataFrame({
        'time': fixed_datasets[t]['time'],
        'train_plugs':plug_preds_train, 
        'train_unplugs':unplug_preds_train, 
        'test_plugs': plug_preds_test, 
        'test_unplugs':unplug_preds_test}).to_csv(f'run_{now}/predictions_fixed_{timeperiods_text[t]}')

### sequential datasets 

for t in range(len(timeperiods)):

    parameter_search_plugs = Optuna(fixed_datasets_DMatrices_plugs[t][0], fixed_datasets_DMatrices_plugs[t][1])
    parameter_search_unplugs = Optuna(fixed_datasets_DMatrices_unplugs[t][0], fixed_datasets_DMatrices_unplugs[t][1])

    optimal_parameters_plugs = parameter_search_plugs.conduct_study('xgb')
    optimal_parameters_unplugs = parameter_search_unplugs.conduct_study('xgb')
    
    xgb_plugs = xgb.XGBRegressor(**optimal_parameters_plugs, objective='reg:squarederror')
    xgb_unplugs = xgb.XGBRegressor(**optimal_parameters_unplugs, objective='reg:squarederror')
    
    for p in range(len(sequential_datasets[t])):

        print('Training model with optimal parameters...')
        xgb_plugs.fit(sequential_datasets_features[t][p][0], sequential_datasets_plugs[t][p][0])
        xgb_unplugs.fit(sequential_datasets_features[t][p][0], sequential_datasets_unplugs[t][p][0])

    # predict target
    print('Predicting...')
    # test datasets
    plug_preds_test = xgb_plugs.predict(fixed_datasets_features[t][1])
    unplug_preds_test = xgb_unplugs.predict(fixed_datasets_features[t][1])
    #train datasets
    plug_preds_train = xgb_plugs.predict(fixed_datasets_features[t][0])
    unplug_preds_train = xgb_unplugs.predict(fixed_datasets_features[t][0])

    # compute results
    print("Computing results...")
    results.append(baseline.evaluate_metrics(fixed_datasets_plugs[t][1], plug_preds_test, 'XGBOOST', 'Plugs', timeperiods_text[t], 'test set'))
    results.append(baseline.evaluate_metrics(fixed_datasets_unplugs[t][1], unplug_preds_test, 'XGBOOST', 'Unplugs', timeperiods_text[t], 'test set'))
    
    results.append(baseline.evaluate_metrics(fixed_datasets_plugs[t][0], plug_preds_train, 'XGBOOST', 'Plugs', timeperiods_text[t], 'train set'))
    results.append(baseline.evaluate_metrics(fixed_datasets_unplugs[t][0], unplug_preds_train, 'XGBOOST', 'Unplugs', timeperiods_text[t], 'train set'))

    print("Saving models and predictions...")
    xgb_plugs_path = f'run_{now}/plugs_sequential_{timeperiods_text[t]}.pkl'
    xgb_unplugs_path = f'run_{now}/unplugs_sequential_{timeperiods_text[t]}.pkl'
    with open(xgb_plugs_path, 'wb') as file:
        pickle.dump(xgb_plugs, file)
    with open(xgb_unplugs_path, 'wb') as file:
        pickle.dump(xgb_unplugs, file)

    pd.DataFrame({
        'time': fixed_datasets[t]['time'],
        'train_plugs':plug_preds_train, 
        'train_unplugs':unplug_preds_train, 
        'test_plugs': plug_preds_test, 
        'test_unplugs':unplug_preds_test}).to_csv(f'run_{now}/predictions_sequential_{timeperiods_text[t]}')
    

pd.DataFrame(results, columns = ['model', 'target', 'timeperiod', 'breakout', 'rsme', 'mae', 'r2']).to_csv(f'run_{now}/results.csv')






