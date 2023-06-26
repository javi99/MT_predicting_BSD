import pandas as pd
import json
import os
import chardet
import numpy as np
import re

csvdata_cols_dropped = ['fecha', 'idBike', 'fleet', 
                        'geolocation_unlock', 'address_unlock', 'locktype', 
                        'unlocktype', 'unlock_station_name', 'lock_station_name',
                        'geolocation_lock', 'address_lock', 'lock_date']

jsondata_cols_dropped =  ['_id', 'user_day_code', 'user_type', 'ageRange', 'zip_code', 'track']


def load_json_bad_format(path):
    '''Formats raw json input into something useable'''

    parsed_data = []

    with open(path, encoding = 'latin-1') as f:
        data = f.read() #data is a full string containing all elements separated by new lines
        for line in data.split("\n"): #we split the string by new lines
            try:
                data_point = json.loads(line) #we load each element as a json to transform the string into a dict object
                parsed_data.append(data_point) #we save it into a list
            except:
                print(line)
    return parsed_data

def add_time_data(df, time_column, extra_col_name = ""):
    """transforms time column into datetime format and adds day, month, year, and hour columns.
    In case there is more than one time column in the dataframe you can extend the name of the
    new variables by using the extra_col_name variable"""
    df[time_column] = pd.to_datetime(df[time_column])
    df[extra_col_name + "day"] = df.apply(lambda row: row[time_column].day, axis = 1)
    df[extra_col_name + "month"] = df.apply(lambda row: row[time_column].month, axis = 1)
    df[extra_col_name + "year"] = df.apply(lambda row: row[time_column].year, axis = 1)
    df[extra_col_name + "hour"] = df.apply(lambda row: row[time_column].hour, axis = 1)

    return df


def station_times(df):

    # create a datetime index from the year, month, day, and hour columns
    df["_id"] = pd.to_datetime(df["_id"])
    df_test = df.copy()

    #we set all the hours to floor. So for example, 00:23:00 will be 00:00:00
    #if we don't do this, the method to detect new rows will not work, because
    #the resample method will change all indices to 01:00:00 like time indices
    # and all indices of the resampled_df will be different from st_df_sample
    df["_id"] = df["_id"].dt.floor("H")
    df['is_new'] = False  # Initialize the new column to False

    duplicates = df[df.duplicated(['_id'], keep=False)]
    if not duplicates.empty:
        print("duplicates detected:")
        print(df_test.iloc[duplicates.index])

    df.drop_duplicates(subset=['_id'], inplace = True)

    # Set the "_id" column as the index of the DataFrame and resample at hourly intervals
    resampled_df = df.set_index("_id").resample('H').ffill()

    # Find the indices of the new rows by comparing the index of the resampled DataFrame
    # with the index of the original DataFrame shifted by one hour
    new_indices = resampled_df.index.difference(df.set_index("_id").index)

    # Set the "is_new" column to True for the new rows
    resampled_df.loc[new_indices, 'is_new'] = True

    return resampled_df

def explode_stations(df):

    # decompress data
    df.reset_index(inplace=True)
    st_df_sample = df.explode("stations").reset_index() #this turns each list into rows with the same 
    time = st_df_sample["_id"]
    is_new = st_df_sample["is_new"]
    st_df_sample = pd.DataFrame(list(st_df_sample["stations"]))

    st_df_sample["time"] = time
    st_df_sample.loc[is_new,"dock_bikes"] = np.NaN
    st_df_sample.loc[is_new,"free_bases"] = np.NaN
    st_df_sample.loc[is_new,"reservations_count"] = np.NaN
    st_df_sample.rename(columns={"id":"id_station"},inplace=True)

    return st_df_sample


def create_stations_df(stations):

    '''
    Creates main stations dataset
    input is a list with the names of the files containing station data
    '''
    
    stations_dataset = pd.DataFrame(columns=['activate', 'name', 'reservations_count', 'light', 'total_bases',
       'free_bases', 'number', 'longitude', 'no_available', 'address',
       'latitude', 'dock_bikes', 'id_station', 'time', 'day', 'month', 'year',
       'hour'])
    
    count = 0
    df_size = 0
    
    for station in stations:

        print(f'processing: {station}')


        # getting full path
        path = data_dl_path + station

        # processing data
        data = load_json_bad_format(path)

        data = pd.DataFrame(data)

        data = station_times(data)

        data = explode_stations(data)

        data = add_time_data(data, "time")

        # adjust datatypes
        data = data.astype({'activate': float,
                    'name': str,
                    'reservations_count': float,
                    'light': float,
                    'total_bases': float,
                    'free_bases': float,
                    'number': str,
                    'longitude': str,
                    'no_available': float,
                    'address': str,
                    'latitude': str,
                    'dock_bikes': float,
                    'id_station': float,
                    'day': int, 
                    'month': int, 
                    'year': int, 
                    'hour': float})
            
        # stack dataset to the previous one
        stations_dataset = pd.concat([stations_dataset, pd.DataFrame(data)], axis=0, ignore_index=True)

        df_size = df_size + len(stations_dataset)

        count = count + 1
    
    print("files processed:", count)
    print("# of rows:", df_size)

    stations_dataset['weekday'] = stations_dataset['time'].apply(lambda x: x.weekday())

    return stations_dataset

def process_movement_csv(path, dtypes, keep_cols):

    # quick clean up of the csv file
    
    csvdata = pd.read_csv(path, sep = ';')
    csvdata.dropna(how='all', inplace = True)

    csvdata = csvdata[keep_cols]

    csvdata = csvdata.astype(dtypes)
    csvdata['unlock_date']  = pd.to_datetime(csvdata['unlock_date'])

    return csvdata

def process_movement_json(path, dtypes, keep_cols):

    '''handles trip data found in the json files'''

    jsondata = load_json_bad_format(path)
    jsondata = pd.DataFrame(jsondata)
    jsondata["_id"] = jsondata["_id"].apply(lambda id: id["$oid"]) # some id values are dictionaries
    # we have noticed that travel time in the json files is seconds so we switch to minutes to align with the csv files
    jsondata['travel_time'] = jsondata['travel_time'] / 60 

    # rename columns to match csv files
    trip_cols = {
    'travel_time': 'trip_minutes',
    'idplug_station': 'station_lock',
    'idunplug_station': 'station_unlock',
    'unplug_hourTime': 'unlock_date'
    }

    jsondata.rename(columns = trip_cols, inplace=True)

    jsondata = jsondata[keep_cols]

    jsondata = jsondata.astype(dtypes)
    
    # date is sometimes a dictionary
    try:
        jsondata['unlock_date']  = pd.to_datetime(jsondata['unlock_date'])
    except:
        jsondata["unlock_date"] = jsondata["unlock_date"].apply(lambda date: date["$date"])
        jsondata['unlock_date']  = pd.to_datetime(jsondata['unlock_date'])

    
    return jsondata 


def create_movements_df(movements):

    '''stacks pandas dataframe created from the csv and json files containing trip data into one master dataframe'''

    movements_dataset = pd.DataFrame(columns=['trip_minutes', 
                                              'station_unlock', 
                                              'station_lock', 
                                              'unlock_date',
                                              'geolocation_unlock', 
                                              'address_unlock', 
                                              'locktype', 
                                              'unlocktype', 
                                              'unlock_station_name', 
                                              'lock_station_name',
                                              'geolocation_lock', 
                                              'address_lock', 
                                              'lock_date'])
    
    
    count = 0
    df_size = 0

    for movement in movements:

        print(f'processing: {movement}')
        
        path = data_dl_path + movement

        trip_dtypes = {
        'trip_minutes': float,
        'station_unlock': str,
        'station_lock': str,
        }

        csvtrip_cols = ['trip_minutes','station_lock',  
                        'station_unlock', 'unlock_date', 
                        'geolocation_unlock', 'address_unlock', 
                        'locktype', 'unlocktype', 
                        'unlock_station_name', 'lock_station_name',
                        'geolocation_lock', 'address_lock', 
                        'lock_date'
                        ]
        
        jsontrip_cols = ['trip_minutes','station_lock',  
                         'station_unlock', 'unlock_date']
        
        if 'csv' in path:
            movement_data = process_movement_csv(path, trip_dtypes, csvtrip_cols)
        elif 'json' in path:
            movement_data = process_movement_json(path, trip_dtypes, jsontrip_cols)

        movement_data = movement_data.replace("",np.NaN)

        df_size = df_size + len(movements_dataset)

        count = count + 1
        
        movements_dataset = pd.concat([movements_dataset, pd.DataFrame(movement_data)], axis=0, ignore_index=True)
    
    print("files processed:", count)
    print("# of rows:", df_size)


    return movements_dataset


### Code starts here ####

data_dl_path = os.getcwd()+'/data/downloading/storage/'

files_dl = os.listdir(data_dl_path)

stations = [file for file in files_dl if 'Usage' not in file and 'movements' not in file and 'trips' not in file and '.DS_Store' not in file] 
print(f"stations files: {set(stations)}")
movements = list(set(files_dl) - set(stations))
print(f"trips files: {set(movements)}")
print(f"stations files: {len(set(stations))}, movements files: {len(set(movements))}")
print(f"total number of files: {len(set(stations)) + len(set(movements))}. Should be 24 per year extracted.")

stations_data = create_stations_df(stations)
print("stations dataframe created")

movements_data = create_movements_df(movements)
print("movements dataframe created")

print(movements_data.head())
stations_data.to_csv(os.getcwd() + '/processing/storage_final/stations_data_raw.csv')
movements_data.to_csv(os.getcwd() + '/processing/storage_final/trips_data_raw.csv')
