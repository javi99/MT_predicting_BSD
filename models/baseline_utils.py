from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import copy
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
import xgboost as xgb



class baseline_model:


    def create_modeling_datasets(self, dataset, timeperiods, train_start):
    # timeperiod is a list of datetime date tuples: (start_date, end_date)

        data = copy.deepcopy(dataset)
    
        train_test_sets = []
    
        for timeperiod in timeperiods:
            test = data[(data['time'] >= timeperiod[0]) & (data['time'] <= timeperiod[1])].loc[:,:]
            train = data[(data['time'] >= train_start) & (data['time'] < timeperiod[0])].loc[:,:]
            train.sort_values('time', inplace = True)
            test.sort_values('time', inplace = True)

            train_test_sets.append([train, test])
    
        return train_test_sets            
                                                     

    def evaluate_metrics(self, target_true, target_predictions, model, target, timeperiod, breakout):
        
        rmse = np.sqrt(mean_squared_error(target_true, target_predictions))
        mae = mean_absolute_error(target_true, target_predictions)
        r2 = r2_score(target_true, target_predictions)
                        
        results_list = [model, target, timeperiod, breakout, rmse, mae, r2]

        return results_list
                        
    def display_feature_importance(self, fitted_model, features):
        fitted_model.feature_importances_

        importances_clf = pd.Series(data = fitted_model.feature_importances_,
                                    index= features)

        importances_sorted_clf = importances_clf.sort_values()

        importances_sorted_clf.plot(kind='barh', color='green')
        plt.title('Gini Features Importances', size=15)
        plt.show()


    def create_feature_datasets(self, datasets, features, frame_type, scaler_type):
        # datasets is a lists of lists:  [train dataframe, test dataframe]
        feature_datasets = copy.deepcopy(datasets)

        if scaler_type == 'standard':
            scaler = StandardScaler()
        if scaler_type == 'minmax':
            scaler = MinMaxScaler()
        
        if frame_type == 'fixed':
            for dataset in feature_datasets:
                dataset[0] = dataset[0][features]
                dataset[1] = dataset[1][features]
                dataset[0] = pd.DataFrame(scaler.fit_transform(dataset[0]), columns = features)
                dataset[1] = pd.DataFrame(scaler.transform(dataset[1]), columns = features)
        
        elif frame_type == 'sequential':
            for dataset in feature_datasets:
                for pair in dataset:
                    pair[0] = pair[0][features]
                    pair[1] = pair[1][features]
                    pair[0] = pd.DataFrame(scaler.fit_transform(pair[0]), columns = features)
                    pair[1] = pd.DataFrame(scaler.transform(pair[1]), columns = features)
        
        return feature_datasets
    
    def create_target_datasets(self, datasets, target, frame_type):
        
        target_datasets = copy.deepcopy(datasets)

        if frame_type == 'fixed':
            for dataset in target_datasets:
                dataset[0] = dataset[0][target]
                dataset[1] = dataset[1][target]   
        
        elif frame_type == 'sequential':
            for dataset in target_datasets:
                for pair in dataset:
                    pair[0] = pair[0][target]
                    pair[1] = pair[1][target]        
        
        return target_datasets
    
    def basic_baseline(self, data):

        baseline_set = copy.deepcopy(data)

        baseline_set.sort_values(['number', 'time'], inplace = True)
        baseline_set.set_index('time', inplace = True)
        baseline_set['pred_unplugs'] =  baseline_set['unplugs_count'].shift(168)
        baseline_set['pred_plugs'] = baseline_set['plugs_count'].shift(168)

        return baseline_set
    
    def stationarity_test(self, data, target, significance_level):
        stationarity_dataset = copy.deepcopy(data)
        stationarity_dataset.sort_values(['time'], inplace = True)
        stationarity_dataset.set_index('time', inplace = True)
        result = adfuller(stationarity_dataset[f"{target}"].values)
        pvalue = result[1]
        if pvalue < significance_level:
            print(f'Augmented Dickey-Fuller test p-value: {pvalue}. The time series is stationary')
        else:
            print(f'Augmented Dickey-Fuller test p-value: {pvalue}. The time series is not stationary')

    def create_weekly_sequential_datasets(self, data):
        window_data = copy.deepcopy(data)

        weekly_sets = []

        for datasets in window_data:
            train_data = datasets[0]
            
            train_data.sort_values(['year', 'week_of_year', 'weekday', 'hour', 'number'], inplace = True)
        
            sequences = []
            
            weeks = len(train_data['year_week_index'].unique())
            
            for week in range(weeks):
                if week + 1 == weeks:
                    break
                else:
                    sequences.append(
                    [train_data[train_data['year_week_index'] == train_data['year_week_index'].unique()[week]].loc[:,:],
                    train_data[train_data['year_week_index'] == train_data['year_week_index'].unique()[week + 1]].loc[:,:]])

            weekly_sets.append(sequences)
            
        
        return weekly_sets
    
    def create_DMatrices(self, feature_datasets, target_datasets, frame_type):
        
        features = copy.deepcopy(feature_datasets)
        targets = copy.deepcopy(target_datasets)

        if frame_type == 'fixed':

            dmatrices =  []

            for n in range(len(features)):
                dtrain = xgb.DMatrix(features[n][0], label=targets[n][0])
                dtest = xgb.DMatrix(features[n][1], label=targets[n][1])
                dmatrices.append([dtrain, dtest])
        
        elif frame_type == 'sequential':
            
            dmatrices = []
            
            for feature in range(len(features)):
                
                batch = []
                
                for pair in range(len(features[feature])):

                    dtrain_plugs = xgb.DMatrix(features[feature][pair][0], label=targets[feature][pair][0])
                    dtest_plugs = xgb.DMatrix(features[feature][pair][1], label=targets[feature][pair][1])
                    batch.append([dtrain_plugs, dtest_plugs])

                dmatrices.append(batch)

        return dmatrices


        





        






    







        





            




        
    
