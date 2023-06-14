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
data_dl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '/storage_final/')

data_dl_path = '/Users/erikagutierrez/Documents/BSE/Term_3/Masters_Thesis/MT_predicting_BSD/processing/storage_final/'

ids = pd.read_csv(data_dl_path + 'bases_bicimad_mod.csv', sep = ';')

# create empty matrices
time_distance_matrix = pd.DataFrame(columns = list(ids['Número'].unique()), index=list(ids['Número'].unique()))
meters_distance_matrix = pd.DataFrame(columns = list(ids['Número'].unique()), index = list(ids['Número'].unique()))

# make list of coordinate pairs

stations = [str(ids.loc[i, 'Latitud'])+','+str(ids.loc[i, 'Longitud']) for i in range(len(ids))]

stations = list(divide_chunks(stations, 10)) # create mini lists of 10 elements each


# Google Maps API endpoint URL
url = "https://maps.googleapis.com/maps/api/distancematrix/json"


partition_end = 10
partition_start = 0
col = 0


for origin in stations:

    time.sleep(60)
    
    for destination in stations:
        # Parameters for the request
        params = {
        "origins": "|".join(origin),
        "destinations": "|".join(destination),
        "key": "{google_api_key}",
        "mode":"bicycling" }

        # Send GET request to the API
        response = requests.get(url, params=params)

        # Parse the JSON response
        data = json.loads(response.text)    

        if data['status'] == 'OK':
            
            
            for x in range(len(data['rows'])):

                time_distance_matrix.iloc[partition_start:partition_end, col + x] = [y['duration']['value'] for y in data['rows'][x]['elements']] # seconds
                meters_distance_matrix.iloc[partition_start:partition_end, col + x] = [y['distance']['value'] for y in data['rows'][x]['elements']] # meters
                
            
            partition_end = partition_end + 10
            partition_start = partition_start + 10

        else:
            print(data['status'])
    
    col = col + 1

   
