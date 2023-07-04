# BiciMAD Bike-Sharing System Dataset Creation and Preliminary Hourly Demand Prediction using Machine and Deep Learning Approaches

**Group:**

- Erika Gutierrez
- Javier Roset Cardona


### Repository Structure 

We have three main folders: data, models, and processing

In each of these folders there is a development folder that contains the notebooks we used to explore the data and test code that would be used in our actual ML pipeline. 


data
- The folder `downloading` contains the code we use to download data from their respective online sources in their raw form.
- The folder `processing` contains the code we use to take this raw data, clean it, and combine it.

processing
- The folder `Feature Engineering` contains the code we used to create the final dataframe of our features and targets. Also contains the script to collect the data used to populate our adjancency matrix. 

models
- contains the various scripts used for creating, training, testing, and assessing our models. The `baseline_modeling.py` contains the code for the baseline and XGBOOST model and `GNN_Main.py` contains the code for the GCNN model. 


