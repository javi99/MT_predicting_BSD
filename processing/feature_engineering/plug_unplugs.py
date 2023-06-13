import pandas as pd
import os

data_dir = '/Users/erikagutierrez/Documents/BSE/Term_3/Masters_Thesis/MT_predicting_BSD/processing/storage_final'

stations = pd.read_csv(data_dir + '/stations_data_final.csv')
trips = pd.read_csv(data_dir + '/trips_data_final.csv')


plugs_df = trips.groupby(["year", "month", "day","hour","mode_id_lock"]).size().reset_index()
plugs_df.columns = ["year", "month", "day","hour","number","plugs_count"]

unplugs_df = trips.groupby(["year", "month", "day","hour","mode_id_unlock"]).size().reset_index()
unplugs_df.columns = ["year", "month", "day","hour","number","unplugs_count"]


stations = pd.merge(stations, plugs_df, on = ['year', 'month', 'day', 'hour', 'number'], how = 'left')
stations = pd.merge(stations, unplugs_df, on = ['year', 'month', 'day', 'hour', 'number'], how = 'left')



stations.plugs_count = stations.plugs_count.fillna(0)
stations.unplugs_count = stations.unplugs_count.fillna(0)


stations.to_csv(data_dir + '/station_plugs.csv')