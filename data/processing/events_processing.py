import chardet
import unicodedata

import pandas as pd
import os
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt

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

def basic_df_load_and_clean(data_folder, data_file_name, sep = None):
    #this 2 lines allow to retrieve the encoding so that we can open the file
    with open(os.path.join(data_folder, data_file_name), 'rb') as f:
        result = chardet.detect(f.read())

    df = pd.read_csv(os.path.join(data_folder, data_file_name), sep=sep,encoding=result['encoding'])

    # define a helper function to remove accents from text
    def remove_accents(text):
        return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

    #setting all column names into lowercase without accents
    # get the current column names
    old_column_names = df.columns.tolist()

    # remove accents and set to lowercase for each column name
    new_column_names = [remove_accents(name).lower() for name in old_column_names]

    # use the new column names to rename the columns
    df = df.rename(columns=dict(zip(old_column_names, new_column_names)))

    # drop rows that have all Nans
    df = df.dropna(how="all")

    return df

data_basepath = "../storage/raw"
dest_path = "../storage/intermediate"
hist_events_path = "general_info/calendario.csv"

hist_events_df = basic_df_load_and_clean(data_basepath, hist_events_path)

# adding year, month, day columnus to events
hist_events_df = add_time_data(hist_events_df, "dia")
hist_events_df.head()

def input_day_type(day):
    if day == "sabado" or day =="domingo":
        out = day
    else:
        out = "laborable"
    return out

# creating binary column to flag non working days
hist_events_df.loc[hist_events_df["laborable / festivo / domingo festivo"].isna(), 
                   "laborable / festivo / domingo festivo"] = hist_events_df.loc[hist_events_df["laborable / festivo / domingo festivo"].isna(), 
                                                                                "laborable / festivo / domingo festivo"].apply(input_day_type)

non_working_labels = ['festivo', 'sabado', 'domingo', 'Festivo']
hist_events_df["work_day_indicator"] = hist_events_df["laborable / festivo / domingo festivo"].apply(
                            lambda value: 0 if value in non_working_labels else 1)


# Adding covid indicator
lock_down_starting_date = datetime(2020, 3, 15)
lock_down_ending_date = datetime(2020, 4, 27)

hist_events_df["covid_indicator"] = hist_events_df["dia"].apply(
                lambda value: 1 if value >= lock_down_starting_date 
                                and value < lock_down_ending_date 
                                else 0)

hist_events_df = hist_events_df[["dia_semana", "day", "month", "year", "work_day_indicator", "covid_indicator"]]
hist_events_df.to_csv(os.path.join(dest_path,"events_final.csv"))