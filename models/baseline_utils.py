from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import copy
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller




class baseline_model:


    def create_modeling_datasets(self, dataset, timeperiods):
    # timeperiod is a list of datetime date tuples: (start_date, end_date)

        data = copy.deepcopy(dataset)
    
        train_test_sets = []
    
        for timeperiod in timeperiods:
            test = data[(data['time'] >= timeperiod[0]) & (data['time'] <= timeperiod[1])]
            train = data[data['time'] < timeperiod[0]]
            train.sort_values('time', inplace = True)
            test.sort_values('time', inplace = True)

            train_test_sets.append([train, test])
    
        return train_test_sets            
                                                     

    def evaluate_metrics(self, target_true, target_predictions, model, target):
        
        rmse = np.sqrt(mean_squared_error(target_true, target_predictions))
        mae = mean_absolute_error(target_true, target_predictions)
        r2 = r2_score(target_true, target_predictions)
                        
        results_list = [model, target, rmse, mae, r2]

        return results_list
                        
    def display_feature_importance(self, fitted_model, fitted_data):
        fitted_model.feature_importances_

        features = fitted_data.columns

        importances_clf = pd.Series(data = fitted_model.feature_importances_,
                                    index= features)

        importances_sorted_clf = importances_clf.sort_values()

        importances_sorted_clf.plot(kind='barh', color='green')
        plt.title('Gini Features Importances', size=15)
        plt.show()


    def create_feature_datasets(self, datasets, features, scaler_type):
        # datasets is a lists of lists:  [train dataframe, test dataframe]
        feature_datasets = copy.deepcopy(datasets)
        
        if scaler_type == 'standard':
            scaler = StandardScaler()
        
        if scaler_type == 'minmax':
            scaler = MinMaxScaler()

        for dataset in feature_datasets:
            dataset[0] = dataset[0][features]
            dataset[1] = dataset[1][features]
            dataset[0] = pd.DataFrame(scaler.fit_transform(dataset[0]), columns = features)
            dataset[1] = pd.DataFrame(scaler.transform(dataset[1]), columns = features)
        
        return feature_datasets
    
    def create_target_datasets(self, datasets, target):
        
        target_datasets = copy.deepcopy(datasets)
       
        for dataset in target_datasets:
            dataset[0] = dataset[0][target]
            dataset[1] = dataset[1][target]    
        
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

    def weekly_window_tuples(self, data, target):
        window_data = copy.deepcopy(data)
        window_data.sort_values(['year', 'week_of_year', 'weekday', 'hour', 'number'], inplace = True)
        training_tuples = []
        weeks = len(window_data['year_week_index'].unique())
        for week in range(weeks):
            if week + 1 == weeks:
                break
            else:
                training_tuples.append(
                (window_data[window_data['year_week_index'] == window_data['year_week_index'].unique()[week]].loc[:,:],
                window_data[window_data['year_week_index'] == window_data['year_week_index'].unique()[week + 1]].loc[:,:]))



        






    







        





            




        
    
