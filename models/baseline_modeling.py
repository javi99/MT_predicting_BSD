import xgboost as xgb
import pandas as pd
import datetime as dt
import os 

os.chdir(os.getcwd() + '/models')

from baseline_model import baseline_model

# get data
data_dir = '/Users/erikagutierrez/Documents/BSE/Term_3/Masters_Thesis/MT_predicting_BSD/processing/storage_final/'

data = pd.read_csv(data_dir + 'bicimad_dataframe.csv')

data.drop(['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'Unnamed: 0_x', 'Unnamed: 0_y'], axis = 1, inplace = True)

data['time'] = pd.to_datetime(data['time'])

# define train test breakouts

forecast_cutoff_start = (max(data['time']) + dt.timedelta(days = -7)) + dt.timedelta(hours = 1) # start date of the last week of data
forecast_cutoff_end = forecast_cutoff_start + dt.timedelta(days = 7) + dt.timedelta(hours = -1) # end date of the last week of data

baseline = baseline_model() # get baseline model class functions

datasets = baseline.create_modeling_datasets(data, [(forecast_cutoff_start, forecast_cutoff_end)]) # result is a list of lists [train, test] for each desired forecast period (need to add validation soon)

# decide what features to use and standardize them s
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

feature_datasets = baseline.create_feature_datasets(datasets, features) # create and standardize the train and test feature datasets  
plug_target_datasets = baseline.create_target_datasets(datasets, target_plugs) # create the train and test plug target vectors
unplug_target_datasets = baseline.create_target_datasets(datasets, target_unplugs) # create the train and test unplug target vectors

# train models

xgb_plugs_model = baseline.train_model(grid_search = False, parameters = {}, model = xgb.XGBRegressor(), X_train = feature_datasets[0][0], y_train = plug_target_datasets[0][0])
xgb_unplugs_model = baseline.train_model(grid_search = False, parameters = {}, model = xgb.XGBRegressor(), X_train = feature_datasets[0][0], y_train = unplug_target_datasets[0][0])

# predict target
plug_preds = xgb_plugs_model.predict(feature_datasets[0][1])
unplug_preds = xgb_unplugs_model.predict(feature_datasets[0][1])

# compute results
results = []
results.append(baseline.evaluate_metrics(plug_target_datasets[0][1], plug_preds, 'XGBOOST'))
results.append(baseline.evaluate_metrics(unplug_target_datasets[0][1], unplug_preds, 'XGBOOST'))

pd.DataFrame(results, columns = ['Model Name', 'RSME', 'MAE', 'R2'])

# compute feature importances

baseline.display_feature_importance(xgb_plugs_model, feature_datasets[0][0])

















    