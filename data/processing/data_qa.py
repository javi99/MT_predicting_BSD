import pandas as pd
import os
import re

print("Loading data...")
stations_data = pd.read_csv(os.getcwd() + '/processing/storage_final/stations_data_raw.csv')
movements_data = pd.read_csv(os.getcwd() + '/processing/storage_final/trips_data_raw.csv')
print("Data loaded")
print("________________")

# adjust data types
print("Adjusting data types...")
stations_data['time'] = pd.to_datetime(stations_data['time'])
print("Time in stations data adjusted")
stations_data['weekday'] = stations_data['time'].apply(lambda x: x.weekday())
print("Weekday added in stations data")

movements_data['unlock_date'] = pd.to_datetime(movements_data['unlock_date'], format='ISO8601')
movements_data['year'] = movements_data['unlock_date'].apply(lambda x: x.year)
print("Years in movements data adjusted")
movements_data['month'] = movements_data['unlock_date'].apply(lambda x: x.month)
print("Months in movements data adjusted")
movements_data['day'] = movements_data['unlock_date'].apply(lambda x: x.day)
print("Days in movements data adjusted")
movements_data['weekday'] = movements_data['unlock_date'].apply(lambda x: x.weekday())
print("Weekday added in movements data")
movements_data['hour'] = movements_data['unlock_date'].apply(lambda x: x.hour)
print("Hours in movements data adjusted")


movements_data['station_unlock'] = movements_data['station_unlock'].map(str)

print("station_unlock in movements data adjusted")

print("Data types adjusted")
print("________________")
# Data Quality code based on results from Data_Quality_Assessment.ipynb

## get number based on name for each id
print("Fixing numbers...")


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

print("Numbers fixed")
print("________________")

print("Inputing outliers of travel time...")
# fill negative trips or trips longer than 5 hours with the median for that month, weekday, and hour

medians = movements_data.groupby(['month', 'weekday', 'hour'])['trip_minutes'].median().reset_index()
medians.rename({'trip_minutes' : 'median_trip_minutes'}, axis = 1, inplace = True)
movements_data = pd.merge(movements_data, medians, on = ['month', 'weekday', 'hour']) 
movements_data[movements_data['trip_minutes'] < 0 ].loc[:,'trip_minutes'] = movements_data['median_trip_minutes']
movements_data[movements_data['trip_minutes'] > 300].loc[:, 'trip_minutes'] = movements_data['median_trip_minutes']

print("Outliers of travel time inputed")
print("________________")
# saving cleaned files
print("Saving files...")
stations_data.to_csv(os.getcwd() + '/processing/storage_final/stations_data_final.csv')
movements_data.to_csv(os.getcwd() + '/processing/storage_final/trips_data_final.csv')
print("Files saved")
