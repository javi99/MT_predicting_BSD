import pandas as pd
import json
import os
import chardet
import numpy as np


def load_json_bad_format(path):

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
    df["_id"] = pd.to_datetime(pd.to_datetime(df["_id"]))
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
    
    stations_dataset = pd.DataFrame(columns=['activate', 'name', 'reservations_count', 'light', 'total_bases',
       'free_bases', 'number', 'longitude', 'no_available', 'address',
       'latitude', 'dock_bikes', 'id_station', 'time', 'day', 'month', 'year',
       'hour'])
    
    count = 0
    df_size = 0
    
    for station in stations:

        print(f'processing: {station}')
        path = data_dl_path + station

        data = load_json_bad_format(path)

        data = pd.DataFrame(data)

        data = station_times(data)

        data = explode_stations(data)

        data = add_time_data(data, "time")

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
                    'day': float, 
                    'month': float, 
                    'year': float, 
                    'hour': float})

        if count != 0:
            newdatasettypes = list(data.dtypes)
            olddatasettypes = list(stations_dataset.dtypes)
            newdatasetcolumns = list(data.columns)
            olddatasetcolumns  = list(stations_dataset.columns)

            if newdatasettypes != olddatasettypes:
                print("Data type mismatch, check file", station)
                print('# old:', len(olddatasettypes))
                print('# new:', len(newdatasettypes))
                for type in range(len(newdatasettypes)):
                    if olddatasettypes[type] != newdatasettypes[type]:
                        print("column old:", olddatasetcolumns[type], olddatasettypes[type])
                        print("column new:", newdatasetcolumns[type], newdatasettypes[type] )
            
            if newdatasetcolumns != olddatasetcolumns:
                print("Data type mismatch, check file", station)
                print('# old:', len(olddatasetcolumns))
                print('# new:', len(newdatasetcolumns))
                for col in range(len(newdatasetcolumns)):
                    if olddatasetcolumns[col] != newdatasetcolumns[col]:
                        print("column old:", olddatasetcolumns[type])
                        print("column new:", newdatasetcolumns[type])
            
        #stations_dataset = pd.concat([stations_dataset, pd.DataFrame(data)], axis=0, ignore_index=True)
        stations_dataset = data
        print(len(stations_dataset))
        df_size = df_size + len(stations_dataset)

        count = count + 1
    
    print("files processed:", count)
    print("# of rows:", df_size)

    return stations_dataset


def create_movements_df(movements):
    movements_dataset = pd.DataFrame(columns=['_id', 
                                              'user_day_code', 
                                              'idplug_base', 
                                              'track', 
                                              'user_type', 
                                              'idunplug_base', 
                                              'travel_time', 
                                              'idunplug_station', 
                                              'ageRange', 
                                              'idplug_station', 
                                              'unplug_hourTime', 
                                              'zip_code'])
    for movement in movements:
        print(f'processing: {movement}')
        
        path = data_dl_path + movement
        
        data = load_json_bad_format(path)

        print(len(data))
        
        movement_data = pd.DataFrame(data)
        movement_data["_id"] = movement_data["_id"].apply(lambda id: id["$oid"])
        #movement_data["unplug_hourTime"] = movement_data["unplug_hourTime"].apply(lambda date: date["$date"] if date is dict else date)
        
        for i in range(len(movement_data)):
            if type(movement_data.loc[i, 'unplug_hourTime']) is dict:
                print(i)
                time = movement_data.loc[i, 'unplug_hourTime']
                movement_data.loc[i, 'unplug_hourTime'] = time["$date"]

        movement_data = movement_data.replace("",np.NaN)
        
        #movements_dataset = pd.concat([movements_dataset, pd.DataFrame(movement_data)], axis=0, ignore_index=True)


    return movements_dataset


### Code starts here ####

#data_dl_path = os.path.join(os.path.dirname(__file__), 'storage')
data_dl_path = os.getcwd()+'/data/downloading/storage/'

files_dl = os.listdir(data_dl_path)

stations = [file for file in files_dl if 'Usage' not in file and 'movements' not in file and 'trips' not in file] 
movements = list(set(files_dl) - set(stations))

stations_data = create_stations_df(stations)

#movements_data = create_movements_df(movements[:3])

#stations_data.to_csv(os.getcwd() + '/processing/storage_final/stations_data.csv')

#movements_data.to_csv(os.getcwd() + '/processing/storage_final/trips_data.csv')



data = pd.read_csv(data_dl_path + movements[2], on_bad_lines='skip', sep=';')

data.dropna(how = 'all', inplace = True)
