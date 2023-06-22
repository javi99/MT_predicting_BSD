from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import copy


class baseline_model():


    def create_modeling_datasets(self, dataset, timeperiods):
    # timeperiod is a list of datetime date tuples: (start_date, end_date)

        data = copy.deepcopy(dataset)
    
        train_test_sets = []
    
        for timeperiod in timeperiods:
            test = data[(data['time'] >= timeperiod[0]) & (data['time'] <= timeperiod[1])]
            train = data[(data['time'] < timeperiod[0]) | (data['time'] > timeperiod[1])]
            train_test_sets.append([train, test])
    
        return train_test_sets            
                    

    def train_model(self, grid_search, parameters, model, X_train, y_train):

        if grid_search == True:
            grid_model = GridSearchCV(estimator=model, param_grid=parameters, cv=5)
            grid_model.fit(X_train, y_train)
            return grid_model
        else:
            model.fit(X_train, y_train)
            return model
                                    

    def evaluate_metrics(self, target_true, target_predictions, model):
        
        rmse = np.sqrt(mean_squared_error(target_true, target_predictions))
        mae = mean_absolute_error(target_true, target_predictions)
        r2 = r2_score(target_true, target_predictions)
                        
        results_list = [model, rmse, mae, r2]

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


    def create_feature_datasets(self, datasets, features):
        # datasets is a lists of lists:  [train dataframe, test dataframe]
        feature_datasets = copy.deepcopy(datasets)
        scaler = StandardScaler()
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
        
    
