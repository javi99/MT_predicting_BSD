import pandas as pd
import datetime as dt
import os


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

data_dir = "../../data/storage/intermediate"

stations = pd.read_csv(data_dir + '/stations_data_final.csv')
trips = pd.read_csv(data_dir + '/trips_data_final.csv')
print("Data loaded")

print("______________")
print("Merging stations ad trips dataframes...")

plugs_df = trips.groupby(["year", "month", "day","hour","mode_id_lock"]).size().reset_index()
plugs_df.columns = ["year", "month", "day","hour","number","plugs_count"]

unplugs_df = trips.groupby(["year", "month", "day","hour","mode_id_unlock"]).size().reset_index()
unplugs_df.columns = ["year", "month", "day","hour","number","unplugs_count"]


stations = pd.merge(stations, plugs_df, on = ['year', 'month', 'day', 'hour', 'number'], how = 'left')
stations = pd.merge(stations, unplugs_df, on = ['year', 'month', 'day', 'hour', 'number'], how = 'left')



stations.plugs_count = stations.plugs_count.fillna(0)
stations.unplugs_count = stations.unplugs_count.fillna(0)
print("Stations and trips merged")

### fix coordinates so that each station has ONLY 1 coordinate in all dataframe
print("______________")
print("Fixing bike stations coordinates...")
numbers_coordinates = stations.groupby('number').apply(
                    lambda group: get_unique_combinations(group[["latitude", "longitude"]])).reset_index()

numbers_coordinates["final_coordinates"] = numbers_coordinates.apply(fix_coordinates, axis = 1)
numbers_coordinates[["latitude", "longitude"]] = numbers_coordinates.apply(
    lambda row: pd.Series([row["final_coordinates"][0], row["final_coordinates"][1]],
                          index=['latitude', 'longitude']),
    axis = 1
    )

stations = stations.drop(["latitude", "longitude"],
                                   axis = 1)
stations = pd.merge(stations, 
                         numbers_coordinates[["number", "latitude", "longitude"]],
                         how="left",
                         on="number")

print("Coordinates fixed")
print("______________")
print("inputing october 2021...")
#### Fixing october 2021 (trips file was corrupted so plugs and unplugs could not be obtained)
hist_stations_df_subset = stations[((stations["year"]==2020) | 
                                    (stations["year"]==2021) | 
                                    (stations["year"]==2022)) &
                                    ((stations["month"]==9) |
                                    (stations["month"]==10) |
                                    (stations["month"]==11))]

# we get the week number and the week day of all hist_stations_df_subset
hist_stations_df_subset["week_number"]=hist_stations_df_subset.apply(lambda row: dt.date(row["year"], row["month"], row["day"]).isocalendar().week, axis = 1)
#weekday is already in the dataframe

# we get the week numbers of the weeks of october 2021
aux_oct_2021 = hist_stations_df_subset.loc[(hist_stations_df_subset["year"]==2021) & (hist_stations_df_subset["month"]==10), ["year","month","day","hour","number","unplugs_count","plugs_count","week_number", "weekday"]]
oct_2021_week_numbers = aux_oct_2021["week_number"].unique()

# we get the same weeks for 2020 and 2022
aux_oct_2020 = hist_stations_df_subset.loc[(hist_stations_df_subset["year"]==2020) & (hist_stations_df_subset["week_number"].isin(oct_2021_week_numbers)), ["hour","number","unplugs_count","plugs_count","week_number", "weekday"]]
aux_oct_2020.columns = ["hour","number","unplugs_count_2020","plugs_count_2020","week_number", "weekday"]
aux_oct_2022 = hist_stations_df_subset.loc[(hist_stations_df_subset["year"]==2022) & (hist_stations_df_subset["week_number"].isin(oct_2021_week_numbers)), ["hour","number","unplugs_count","plugs_count","week_number", "weekday"]]
aux_oct_2022.columns = ["hour","number","unplugs_count_2022","plugs_count_2022","week_number", "weekday"]

# we merge everything
aux_oct_2020_2021 = pd.merge(aux_oct_2021, aux_oct_2020, on=["week_number","weekday","hour", "number"], how="left")
aux_oct_2020_2022 = pd.merge(aux_oct_2020_2021, aux_oct_2022, on=["week_number", "weekday", "hour", "number"], how="left")

# filling nans of 2020. Reason explained in jupyter notebook
aux_oct_2020_2022.loc[aux_oct_2020_2022["unplugs_count_2020"].isna(), "unplugs_count_2020"] = aux_oct_2020_2022.loc[aux_oct_2020_2022["unplugs_count_2020"].isna(), "unplugs_count_2022"]
aux_oct_2020_2022.loc[aux_oct_2020_2022["plugs_count_2020"].isna(), "plugs_count_2020"] = aux_oct_2020_2022.loc[aux_oct_2020_2022["plugs_count_2020"].isna(), "plugs_count_2022"]

# interpolating
aux_oct_2020_2022["unplugs_count"] = (aux_oct_2020_2022["unplugs_count_2022"] + aux_oct_2020_2022["unplugs_count_2020"])/2
aux_oct_2020_2022["plugs_count"] = (aux_oct_2020_2022["plugs_count_2022"] + aux_oct_2020_2022["plugs_count_2020"])/2

#plugs and unplugs have to be ints
aux_oct_2020_2022["unplugs_count"] = aux_oct_2020_2022["unplugs_count"].map(int)
aux_oct_2020_2022["plugs_count"] = aux_oct_2020_2022["plugs_count"].map(int)

oct_2021 = aux_oct_2020_2022.loc[:,["month","day","hour","number","unplugs_count","plugs_count"]]
oct_2021["year"] = 2021

# Merge the two dataframes based on the common columns
stations = stations.merge(oct_2021, on=['year', 'month', 'day', 'hour', 'number'], suffixes=('', '_replacement'), how="left")

# Select the columns to keep from the replacement dataframe and drop the original columns
stations['plugs_count'] = stations['plugs_count_replacement'].fillna(stations['plugs_count'])
stations['unplugs_count'] = stations['unplugs_count_replacement'].fillna(stations['unplugs_count'])
stations = stations.drop(['plugs_count_replacement', 'unplugs_count_replacement'], axis=1)

print("October 2021 inputed")
print("______________")
print("Inputing dock_bikes, free_bases and reservation_counts...")

# inputing dock_bikes and free_bases
dock_bikes_null_indices = stations[stations["dock_bikes"].isna()].index
for index in dock_bikes_null_indices:
    stations.loc[index, "dock_bikes"] = stations.loc[index-1, "dock_bikes"] + stations.loc[index, "plugs_count"]\
                                                -stations.loc[index, "unplugs_count"]
    stations.loc[index, "free_bases"] = stations.loc[index, "total_bases"] - stations.loc[index, "dock_bikes"] 

# inputing reservations_count
stations["reservations_count"] = stations["reservations_count"].fillna(0)

print("dock_bikes, free_bases and reservation_counts inputed")
print("______________")
print("saving station_plugs.csv...")

stations.to_csv(os.path.join(data_dir,"station_plugs.csv"))
print("station_plugs.csv saved")
