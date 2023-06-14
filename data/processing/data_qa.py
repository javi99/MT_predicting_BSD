import pandas as pd
import os
import re

stations_data = pd.read_csv(os.getcwd() + '/processing/storage_final/stations_data_raw.csv')

movements_data = pd.read_csv(os.getcwd() + '/processing/storage_final/trips_data_raw.csv')

# adjust data types

movements_data['unlock_date'] = pd.to_datetime(movements_data['unlock_date'], format='ISO8601')
movements_data['year'] = movements_data['unlock_date'].apply(lambda x: x.year)
movements_data['month'] = movements_data['unlock_date'].apply(lambda x: x.month)
movements_data['day'] = movements_data['unlock_date'].apply(lambda x: x.day)
movements_data['weekday'] = movements_data['unlock_date'].apply(lambda x: x.weekday())
movements_data['hour'] = movements_data['unlock_date'].apply(lambda x: x.hour)

stations_data['time'] = pd.to_datetime(stations_data['time'])
stations_data['weekday'] = stations_data['time'].apply(lambda x: x.weekday())


for station in range(len(movements_data['station_unlock'])):
    try:
        movements_data.loc[station, 'station_unlock'] = float(movements_data.loc[station, 'station_unlock'])
    except:
        pass


# Data Quality code based on results from Data_Quality_Assessment.ipynb

## get number based on name for each id

lock_ids = []

for index in range(len(movements_data)):
    scrape = re.findall('\d+[a-z]? -', str(movements_data.loc[index, 'lock_station_name']))
    if scrape:
        lock_ids.append(scrape[0].replace('-', '').strip())
    else:
        lock_ids.append(None)

movements_data['scraped_station_lock'] = lock_ids

unlock_ids = []

for index in range(len(movements_data)):
    scrape = re.findall('\d+[a-z]? -', str(movements_data.loc[index, 'unlock_station_name']))
    if scrape:
        unlock_ids.append(scrape[0].replace('-', '').strip())
    else:
        unlock_ids.append(None)

movements_data['scraped_station_unlock'] = unlock_ids


# get mode number by station id
unlock_station_mode = movements_data.groupby('station_unlock')['scraped_station_unlock'].agg(lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else None).reset_index()
lock_station_mode = movements_data.groupby('station_lock')['scraped_station_lock'].agg(lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else None).reset_index()

unlock_station_mode.rename({'scraped_station_unlock':'mode_id_unlock'}, axis = 1, inplace = True)
lock_station_mode.rename({'scraped_station_lock':'mode_id_lock'}, axis = 1, inplace = True)

# merge the mode number to trips dataset
movements_data = pd.merge(movements_data, unlock_station_mode, on = 'station_unlock')
movements_data = pd.merge(movements_data, lock_station_mode, on = 'station_lock')


# fill negative trips or trips longer than 5 hours with the median for that month, weekday, and hour

medians = movements_data.groupby(['month', 'weekday', 'hour'])['trip_minutes'].median().reset_index()
medians.rename({'trip_minutes' : 'median_trip_minutes'}, axis = 1, inplace = True)
movements_data = pd.merge(movements_data, medians, on = ['month', 'weekday', 'hour']) 
movements_data[movements_data['trip_minutes'] < 0 ].loc[:,'trip_minutes'] = movements_data['median_trip_minutes']
movements_data[movements_data['trip_minutes'] > 300].loc[:, 'trip_minutes'] = movements_data['median_trip_minutes']


stations_data.to_csv(os.getcwd() + '/processing/storage_final/stations_data_final.csv')

movements_data.to_csv(os.getcwd() + '/processing/storage_final/trips_data_final.csv')
