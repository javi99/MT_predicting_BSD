import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

os.chdir(os.getcwd() + '/models')

from baseline_utils import baseline_model


def get_model(pickle_obj, folder):

    path = os.getcwd() + folder + pickle_obj

    with open(path, 'rb') as file:
        model = pickle.load(file)
    
    return model


plugs_model_2022 = get_model('/plugs_fixed_all_of_2022.pkl', '/trainings/run_03-07-2023_15:55:54')
unplugs_model_2022 = get_model('/unplugs_fixed_all_of_2022.pkl', '/trainings/run_03-07-2023_15:55:54')


plugs_model_INNOVA = get_model('/plugs_fixed_INNOVA_comparison.pkl', '/trainings/run_03-07-2023_15:55:54')
unplugs_model_INNOVA = get_model('/unplugs_fixed_INNOVA_comparison.pkl', '/trainings/run_03-07-2023_15:55:54')




baseline_model().display_feature_importance(plugs_model_2022, plugs_model_2022.feature_names_in_)
baseline_model().display_feature_importance(unplugs_model_2022, unplugs_model_2022.feature_names_in_)
baseline_model().display_feature_importance(plugs_model_INNOVA, plugs_model_INNOVA.feature_names_in_)
baseline_model().display_feature_importance(unplugs_model_INNOVA, unplugs_model_INNOVA.feature_names_in_)

print('2022 plugs params:', plugs_model_2022.get_params())
print('2022 unplugs params:', unplugs_model_2022.get_params())

print('INNOVA plug params:', plugs_model_INNOVA.get_params())
print('INNOVA unplug params:', unplugs_model_INNOVA.get_params())
