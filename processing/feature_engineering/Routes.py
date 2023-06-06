import pandas as pd
import os 
import requests
import json

data_dl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '/storage_final/')

data_dl_path = '/Users/erikagutierrez/Documents/BSE/Term_3/Masters_Thesis/MT_predicting_BSD/processing/storage_final/'

ids = pd.read_csv(data_dl_path + 'bases_bicimad_mod.csv', sep = ';')

stations = [str(ids.loc[i, 'Latitud'])+','+str(ids.loc[i, 'Longitud']) for i in range(len(ids))]

# Google Maps API endpoint URL
url = "https://maps.googleapis.com/maps/api/distancematrix/json"

# Parameters for the request
params = {
    "origins": "|".join(stations),
    "destinations": "|".join(stations),
    "key": "",
    "travelMode":"TWO_WHEELER",
}

# Send GET request to the API
response = requests.get(url, params=params)

# Parse the JSON response
data = json.loads(response.text)

# Print the route matrix data
print(json.dumps(data, indent=4))



