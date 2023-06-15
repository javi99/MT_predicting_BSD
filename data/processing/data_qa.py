import pandas as pd
import os
import re

stations_data = pd.read_csv(os.getcwd() + '/processing/storage_final/stations_data_raw.csv')

movements_data = pd.read_csv(os.getcwd() + '/processing/storage_final/trips_data_raw.csv')

def get_unique_combinations(group):
    unique_combinations = group.drop_duplicates().values.tolist()
    combination_count = len(unique_combinations)
    return pd.Series([unique_combinations, combination_count], index=['UniqueCombinations', 'CombinationCount'])

def fix_coordinates(row):
    if corrected_coordinates.get(row["number"]) is not None:
        return corrected_coordinates[row["number"]]
    return(row["UniqueCombinations"][0])

corrected_coordinates={
        "104":[40.426474003597164,-3.6898352178135774],
        "116a":[40.42387800408879,-3.7121656473771725],
        "116b":[40.42387800408879,-3.7121656473771725],
        "124":[40.43082800000002,-3.69926999999999],
        "165":[40.39237946326748,-3.6974897974164445],
        "173":[40.39662006380033,-3.701471192778041],
        "178":[40.403088000000004,-3.678756999999999],
        "211":[40.42004007293344,-3.7039815654211616],
        "215":[40.43072055955257,-3.662954888947103],
        "261":[40.45112066550112,-3.729695630322434],
        "21a":[40.420623917597965,-3.7000545954465824],
        "21b":[40.420623917597965,-3.7000545954465824],
        "151":[40.44420178746236,-3.6899872845471027],
        "153":[40.44420178746236,-3.6899872845471027],
        "189":[40.434895,-3.6555399999999993],
        "65":[40.4189241163882,-3.681836419670279],
        "33":[40.418380277533785,-3.6928282726731543],
        "260":[40.44717541368264,-3.7166395793647733],
        "1a":[40.42151116612533,-3.704362241173922],
        "1b":[40.42151116612533,-3.704362241173922],
        "53":[40.407131275247245,-3.7055071717752286],
        "131":[40.43171641183384,-3.6923551873869287],
        "28":[40.43171641183384,-3.6923551873869287],
        "140":[40.43171641183384,-3.6923551873869287],
        "38":[40.41969414751022,-3.6951017694181787],
        "86":[40.41969414751022,-3.6951017694181787],
        "76":[40.41395994677364,-3.685853512734829],
        "110":[40.434629000000015,-3.7204379999999992],
        "23":[40.42309037125906,-3.700477250124359],
        "94":[40.42309037125906,-3.700477250124359],
        "25a":[40.41651026888697,-3.695726302314537],
        "25b":[40.41651026888697,-3.695726302314537],
        "111":[40.425913792284845,-3.717928714794998],
        "83":[40.41328,-3.674498000000006],
        "161":[40.42155,-3.721318000000009],
        "97":[40.41570900646776,-3.6877929143310473],
        "98":[40.41570900646776,-3.6877929143310473],
        "122":[40.421733608598586,-3.6879645757080004],
        "36":[40.41870256867726,-3.7085154673098475],
        "237":[40.438343068707454,-3.6986021144169112],
        "24":[40.422865618651684,-3.705213976877948],
        "209":[40.453151941777655,-3.6709956094532226],
        "10":[40.425388295351915,-3.690128993099775],
        "37":[40.417138095218945,-3.7042583161474174],
        "105":[40.41764282648464,-3.7005307372920337],
        "130":[40.433004497510204,-3.7065798341832257],
        "129":[40.433004497510204,-3.7065798341832257],
        "78":[40.40369045600693,-3.6806467600737247],
        "15":[40.4262383, -3.7074453],
        "103":[40.43037, -3.68653],
        "100":[40.42478, -3.67384],
        "5":[40.42852, -3.70205],
        "101":[40.4231526, -3.6691524]
        }


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

# fix coordinates so that each station has ONLY 1 coordinate in all dataframe

numbers_coordinates = stations_data.groupby('number').apply(
                    lambda group: get_unique_combinations(group[["latitude", "longitude"]])).reset_index()

numbers_coordinates["final_coordinates"] = numbers_coordinates.apply(fix_coordinates, axis = 1)
numbers_coordinates[["latitude", "longitude"]] = numbers_coordinates.apply(
    lambda row: pd.Series([row["final_coordinates"][0], row["final_coordinates"][1]],
                          index=['latitude', 'longitude']),
    axis = 1
    )

stations_data = stations_data.drop(["latitude", "longitude"],
                                   axis = 1)
stations_data = pd.merge(stations_data, 
                         numbers_coordinates[["number", "latitude", "longitude"]],
                         how="left",
                         on="number")

# saving cleaned files
stations_data.to_csv(os.getcwd() + '/processing/storage_final/stations_data_final.csv')
movements_data.to_csv(os.getcwd() + '/processing/storage_final/trips_data_final.csv')
