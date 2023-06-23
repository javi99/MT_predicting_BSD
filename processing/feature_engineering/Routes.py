import pandas as pd
import os 
import requests
import json
import time
from credentials import google_api_key

#For computing a route matrix, the rate limit is 3,000 elements 
# (number of origins × number of destinations) per minute and maximum 100 elements per server-side request.
# so we create the following function

def divide_chunks(l, n):
    '''to break list into equal sized chunks'''
    for i in range(0, len(l), n): 
        yield l[i:i + n]

# get data
#data_dl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '/storage_final/')

data_dl_path = '/Users/javier/Desktop/TFM/MT_predicting_BSD/modelling/developing/'

ids = pd.read_csv(data_dl_path + 'numbers_with_coordinates_and_indices.csv', sep = ',')

window = 10
#we order the coordinates in terms of the index column, so that later on in the matrix we can identify
#each distance with the corresponding stations
ids = ids.sort_values("index")

# create empty matrices
time_distance_matrix = pd.DataFrame(columns = list(ids['index'].unique()), index=list(ids['index'].unique()))
meters_distance_matrix = pd.DataFrame(columns = list(ids['index'].unique()), index = list(ids['index'].unique()))

# make list of coordinate pairs

#stations has the same order as ids
stations = [str(ids.loc[i, 'latitude'])+','+str(ids.loc[i, 'longitude']) for i in range(len(ids))]
stations = list(divide_chunks(stations, window)) # create mini lists of 10 elements each

# Google Maps API endpoint URL
url = "https://maps.googleapis.com/maps/api/distancematrix/json"


for origin in range(len(stations)):
    print(f"iteration {origin+1} of {len(stations)} iterations")

    partition_start_origin = origin*window
    partition_end_origin = partition_start_origin+window
    if partition_end_origin > len(ids):
        partition_end_origin = len(ids)
        
    origins = stations[origin]

    for destination in range(len(stations)):
        partition_start_dest = destination*window
        partition_end_dest = partition_start_dest+window
        if partition_end_dest > len(ids):
            partition_end_dest = len(ids)
        
        destinations = stations[destination]

        # Parameters for the request
        params = {
        "origins": "|".join(origins),
        "destinations": "|".join(destinations),
        "key": f"{google_api_key}",
        "mode":"walking" }

        # Send GET request to the API
        response = requests.get(url, params=params)
        
        # Parse the JSON response
        data = json.loads(response.text)    
        if data['status'] == 'OK':  

            for origin_processed in range(len(data['rows'])):

                time_distance_matrix.iloc[partition_start_origin+origin_processed, partition_start_dest:partition_end_dest] = [y['duration']['value'] for y in data['rows'][origin_processed]['elements']] # seconds
                meters_distance_matrix.iloc[partition_start_origin+origin_processed, partition_start_dest:partition_end_dest] = [y['distance']['value'] for y in data['rows'][origin_processed]['elements']] # meters

        else:
            print(data['status'])
        
        #time.sleep(30)

meters_distance_matrix.to_csv("meters_distance_matrix_walking.csv", index=False)
time_distance_matrix.to_csv("time_distance_matrix_walking.csv", index=False)

   
