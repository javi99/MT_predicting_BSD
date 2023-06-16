"""This script will put together all the weather data files into 1 big
file prepared for merging with the other dataframes (stations, trips and events)"""

import pandas as pd
import os
import chardet
import unicodedata
import datetime

from sklearn.neighbors import NearestCentroid

import seaborn as sns
import matplotlib.pyplot as plt

data_folder = "../storage/raw"
destination_folder = "../storage/intermediate"
general_weather_file = "general_info/Estaciones_control_datos_meteorologicos.csv"

hist_weather_folder = os.path.join(data_folder, 
                                   "historical_data/weather_data")
hist_stations_folder = os.path.join(data_folder, 
                                    "historical_data/stations_data")

# this file should be the definitive merged file between stations and trips. This will be used only
# to test the code
hist_bike_stations = "../storage/intermediate/station_plugs.csv"

days_per_month = {
    1:31,
    2:28,
    3:31,
    4:30,
    5:31,
    6:30,
    7:31,
    8:31,
    9:30,
    10:31,
    11:30,
    12:31
}


def basic_df_load_and_clean(folder, data_file_name, sep = None):
    #this 2 lines allow to retrieve the encoding so that we can open the file
    with open(os.path.join(folder, data_file_name), 'rb') as f:
        result = chardet.detect(f.read())

    df = pd.read_csv(os.path.join(folder, data_file_name), 
                     sep=sep,encoding=result['encoding'])

    # define a helper function to remove accents from text
    def remove_accents(text):
        return ''.join(c for c in unicodedata.normalize('NFD', text) 
                       if unicodedata.category(c) != 'Mn')

    #setting all column names into lowercase without accents
    # get the current column names
    old_column_names = df.columns.tolist()

    # remove accents and set to lowercase for each column name
    new_column_names = [remove_accents(name).lower() for name in 
                        old_column_names]

    # use the new column names to rename the columns
    df = df.rename(columns=dict(zip(old_column_names, new_column_names)))

    # drop rows that have all Nans
    df = df.dropna(how="all")

    return df

def parse_hour(hour):
    if "HO" in hour:
        out = hour.replace("HO","")
    if "H" in hour:
        out = hour.replace("H","")
    out = int(out)
    if out == 24:
        out = 0
    return out

def complete_weather_dataframe(folder):
    # getting all filenames
    filenames = [filename for filename in os.listdir(folder)]
    filenames = list(filter(lambda filename: filename != ".DS_Store",
                            filenames))

    # definitive dataframe
    definitive_weather_df = pd.DataFrame()

    #loop through all files
    for filename in filenames:
        # load dataframe
        path = os.path.join(folder, filename)
        df = pd.read_csv(path, sep=";")

        #concatenate it to the definitive dataframe
        definitive_weather_df = pd.concat([definitive_weather_df, df])
    
    return definitive_weather_df

### 1- LOADING DATA ###
# here we will need to put together the weather data
weather_stations_info_df = basic_df_load_and_clean(data_folder, 
                                                   general_weather_file, 
                                                   sep = ";")
hist_stations_df = pd.read_csv(hist_bike_stations)

hist_weather_df = complete_weather_dataframe(hist_weather_folder)

# From historical bike stations information we will want a subset of the dataframe with only 1 row per station, 
# and only the columns id, longitude and magnitude so that we can relate bike stations with weather stations
bs_df = hist_stations_df.groupby("number").first().reset_index()[["number", "longitude", "latitude"]]
bs_df.columns = ["number", "longitud", "latitud"]

print("LOADING: DONE")
### 2- CLEANING HIST WEATHER DATA ###

# 1st problem: some stations doesnt have all days registered
# in all months. Example: in a month of 31 days, station 108 had
# only 30 days registered.

# 2nd problem: Incoherences between weather stations register and
#  weather stations gathering data: some weather stations exist in the
# general info file but do not exist in the historical registering.

# we ensure we are only working with the stations that are gathering data at any time
print("________________________________")

# we get the number of days registered per station per magnitude per year per month
df_stations_registers = hist_weather_df[["ESTACION","MAGNITUD","ANO","MES", "DIA"]].groupby(["ESTACION","MAGNITUD","ANO","MES"]).count().reset_index()
df_stations_registers["time"] =df_stations_registers.apply(lambda row: datetime.datetime(row["ANO"], row["MES"],1), axis = 1)

# we introduce a column indicating the number of days each month has
df_stations_registers["month_days"] = df_stations_registers.apply(lambda row: days_per_month[row["MES"]], axis = 1)
# Then we filter each row to eliminate all data points not registering all days of the month
df_stations_registers_filtered = df_stations_registers[df_stations_registers["month_days"] <= df_stations_registers["DIA"]]
# finally we get only the stations that register a magnitude for the whole month per each month
df_stations_registers_filtered = df_stations_registers_filtered[["ESTACION", "MAGNITUD", "ANO", "MES"]]
hist_weather_df_filtered = pd.merge(df_stations_registers_filtered,hist_weather_df, on=["ESTACION", "MAGNITUD", "ANO", "MES"], how="left")

# we add the latitude and longitude to all weather stations in ws_df_filtered
weather_stations_info_df_filtered=pd.merge(df_stations_registers_filtered, weather_stations_info_df[["codigo_corto","longitud", "latitud"]], left_on="ESTACION", right_on="codigo_corto")

print("CLEANING: DONE")

MAGNITUDES = [81, 82, 83, 86, 87, 88, 89]

### 2- MERGING MECHANISM ###

# 1st: Assigning weather station to bike stations for each magnitude

# from bike stations, we get a list of 1 row per
# station with only id, latitude and longitude
bs_df = hist_stations_df.groupby("number").first().reset_index()
bs_df = bs_df[["number", "longitude", "latitude"]]
bs_df.columns = ["number", "longitud", "latitud"]

### 2.1 - Assigning weather station to bike stations for each magnitude ###
# adding for each magnitude the nearest weather station to each bike station per year per month
print("________________________________")
print("Building relation between weather stations per and bike stations...")
classifier = NearestCentroid()

bs_df = bs_df.sort_values("number")
# we have a df of per each month of each year and each mangitude, which stations register the whole month.
# lets assign to those stations the nearest bike station at each moment in time.
coord_bs = bs_df[["latitud", "longitud"]]
relate_ws_bs = pd.DataFrame(columns=["number","longitud","latitud", "ANO", "MES","MAGNITUD", "weather_station"])

for year in range(2019,2024):
    for month in range(1,13):
        print(f"Building relation: year {year}, month {month}")
        if year == 2023 and month > 4:
            break
        
        for magnitude in MAGNITUDES:
            
            #we select the rows of the stations of that month
            ws_df_magnitude = weather_stations_info_df_filtered[
                    (weather_stations_info_df_filtered["ANO"] == year) & 
                    (weather_stations_info_df_filtered["MES"] == month) & 
                    (weather_stations_info_df_filtered["MAGNITUD"] == magnitude)
                    ]

            #we get the coordinates of the stations that measure that specific magnitude
            coord_ws_magnitude = ws_df_magnitude[["latitud","longitud"]]
        
            stations_magnitude = ws_df_magnitude["ESTACION"]
           
            # we fit the model with the stations of this magnitude
            classifier.fit(coord_ws_magnitude, stations_magnitude)
            # we assign a weather station to each bike station per each magnitude in that moment of time
            ws_for_bs_magnitude = classifier.predict(coord_bs)
            
            bs_df["ESTACION"] = ws_for_bs_magnitude

            # we get the bike station in the weather dataframe. CAREFUL! not all weather stations
            # will have a bike stations BUT all bike stations will have a weather station
            merg = pd.merge(ws_df_magnitude[["ANO", "MES","MAGNITUD", "ESTACION"]],bs_df, on="ESTACION", how="left")
            # we filter through the weather stations that HAVE a bike station
            merg = merg.loc[~merg["number"].isna(), ["number","longitud","latitud", "ANO", "MES","MAGNITUD", "ESTACION"]]
            
            merg.columns = ["number","longitud","latitud", "ANO", "MES","MAGNITUD", "weather_station"]
            # we put all the information into a final dataframe
            relate_ws_bs = pd.concat([relate_ws_bs, merg])


relate_ws_bs = relate_ws_bs.pivot(index=['ANO', 'MES', 'number',"longitud","latitud"], columns='MAGNITUD', values='weather_station').reset_index()
relate_ws_bs.columns = ['ANO', 'MES', 'number',"longitud","latitud", 'weather_station_81', 'weather_station_82', 
                  'weather_station_83', 'weather_station_86', 'weather_station_87', 'weather_station_88',
                  'weather_station_89']

### 3- PROCESSING WEATHER DATA ###
print("________________________________")
print("Starting processing of weather data")

melted_df = pd.melt(hist_weather_df_filtered, id_vars=['ESTACION', 'MAGNITUD', 'PUNTO_MUESTREO', 'ANO', 'MES', 'DIA'],
                        value_vars=['H{:02d}'.format(hour) for hour in range(1, 25)],
                        var_name='HORA', value_name='CANTIDAD')


hist_weather = pd.DataFrame(columns=melted_df.columns)
hist_weather.insert(0, "number", None)

print("Building historical dataframe...")

for year in range(2019,2024):

    aux_year = pd.DataFrame(columns=melted_df.columns)
    aux_year.insert(0, "number", None)

    for month in range(1,13):
        print(f"Building historical: year {year}, month {month}")
        if year == 2023 and month > 4:
            break

        aux_month = pd.DataFrame(columns=melted_df.columns)
        aux_month.insert(0, "number", None)
        
        for magnitude in MAGNITUDES:
            bs_df_column = "weather_station_" + str(magnitude)
            # we get the assigned weather stations for each magnitude (we don't need the rest)
            magnitude_stations = relate_ws_bs.loc[(relate_ws_bs["ANO"]==year) & (relate_ws_bs["MES"]==month),bs_df_column].unique()
            # we filter historical weather station data with the assigned codes
            hist_weather_magnitude_stations = melted_df[(melted_df["ANO"]==year) &
                                             (melted_df["MES"]==month)&
                                             (melted_df["MAGNITUD"]==magnitude)&
                                             (melted_df["ESTACION"].isin(magnitude_stations))]
            # we subset the bike stations information df so that we only merge the current magnitude to the hist weather df
            bs_df_magnitude = relate_ws_bs.loc[(relate_ws_bs["ANO"]==year) & (relate_ws_bs["MES"]==month),["number", bs_df_column]]
            
            # we merge the weather station to each bike station for each moment in time and for each magnitude, as depending
            # on this information each bike station will get the information from one weather station or another
            aux = bs_df_magnitude.merge(hist_weather_magnitude_stations, how = "left", right_on= ["ESTACION"], left_on=bs_df_column)
            aux = aux.drop(bs_df_column, axis = 1)
            
            aux_month = pd.concat([aux_month, aux])

        aux_year = pd.concat([aux_year, aux_month])
    
    hist_weather = pd.concat([hist_weather, aux_year])

hist_weather = hist_weather.pivot_table(index=['number', 'ANO', 'MES', 'DIA', 'HORA'], columns='MAGNITUD', values='CANTIDAD').reset_index()
hist_weather["HORA"] = hist_weather["HORA"].apply(parse_hour)
#we want all number values to be strings
hist_weather["number"] = hist_weather["number"].map(str)
hist_weather.columns = ["number", "year", "month", "day", "hour", 81, 82, 83, 86, 87, 88, 89]
print(hist_weather.head())

print("SAVING FINAL WEATHER DATAFRAME")
hist_weather.to_csv(os.path.join(destination_folder, "weather_final.csv"), index=False)
