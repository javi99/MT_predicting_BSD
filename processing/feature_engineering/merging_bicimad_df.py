import pandas as pd
import os
data_dir = "../../data/storage/intermediate"

print("Loading weather dataframe...")
weather_df = pd.read_csv(os.path.join(data_dir,"weather_final.csv"))
print("Weather dataframe loaded")
print("Loading station_plugs dataframe...")
hist_stations_df = pd.read_csv(os.path.join(data_dir,"station_plugs.csv"))
print("Weather dataframe loaded")
print("______________")
print("Merging stations and trips with weather...")

weather_df["number"] = weather_df["number"].map(str)
stations_weather = pd.merge(hist_stations_df, weather_df, 
         how="left", 
         on=["number", "year", "month", "day", "hour"])

print("Stations and trips merged with weather")

print("______________")
print("Merging events...")

events = pd.read_csv("../../data/storage/intermediate/events_final.csv")
definitive_df = stations_weather.merge(events, on = ["year", "month", "day"], how="left")

print("Events merged")

print("______________")
print("Saving bicimad_dataframe.csv...")

definitive_df.to_csv(data_dir + '/bicimad_dataframe.csv')

print("bicimad_dataframe.csv saved")