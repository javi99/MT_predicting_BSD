import pandas as pd
import re
import requests
import os
import time
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager #details: https://pypi.org/project/webdriver-manager/
from dateutil import rrule
from datetime import datetime
import zipfile

#data_dl_path = os.path.join(os.path.dirname(__file__), 'stations_raw')
data_dl_path = os.getcwd()+'/data/downloading/stations_raw'


start_date = datetime(2018, 7, 1)
end_date = datetime(2021, 6, 30)

date_range = list(rrule.rrule(rrule.MONTHLY, dtstart=start_date, until=end_date))

months = []
years = []
for date in date_range:
    months.append(date.strftime("%B"))
    years.append(date.strftime("%Y"))

months_translator = {'January':'enero',
                     'February': 'febrero',
                     'March': 'marzo',
                     'April': 'abril',
                     'May': 'mayo',
                     'June': 'junio',
                     'July': 'julio',
                     'August':'agosto',
                     'September':'septiembre',
                     'October': 'octubre', 
                     'November': 'noviembre',
                     'December': 'diciembre'}

months = [months_translator[month] for month in months]


options = webdriver.ChromeOptions()
prefs = {"download.default_directory" : data_dl_path}
options.add_experimental_option("prefs",prefs)

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

driver.get('https://opendata.emtmadrid.es/Datos-estaticos/Datos-generales-(1)')


links = driver.find_elements(By.TAG_NAME, 'a')

hrefs= []
titles = []
for link in links:
    link_title = link.get_attribute('title').lower()
    if link.get_attribute('target') == '_blank' and 'enlace' not in link_title:
        for i in range(len(months)):
            if f'{months[i]} {years[i]}' in link_title or f'{months[i]} de {years[i]}' in link_title:
               hrefs.append(link.get_attribute('href'))
               titles.append(link_title)
               link.click()
               time.sleep(5)

time.sleep(30)

driver.close()


# move file name to dl file name match to into for loop?

print(hrefs)
print(titles)
current_time = datetime.now().time()

pd.DataFrame({'title':titles, 'links':hrefs}).to_csv(f'{data_dl_path}/download_{current_time}.csv')

# unzip files
zip_path = "/path/to/archive.zip"

# Specify the path to the directory where files will be extracted
extract_path = "/path/to/extract"

#match files to their href

dl_files = []
for href in hrefs:
    p = re.search('(?<=\/)[^\/]+?(?=\.aspx)', href)
    dl_files.append(p.group(0))

files_dl = os.listdir(data_dl_path)




def ziporrar(file):
    if '.zip' in file or '.rar' in file:
        return True
    else:
        return False

def unzipper(files):
    for file in files:
        if ziporrar(file):
            print('ok')


    # 1. check if file is .zip or .rar or other
    # 2. unzip file and put contents into extract folder (same name as title?)
    # 3. delete .zip and .rar files




'''
# Create the extract directory if it doesn't exist
if not os.path.exists(extract_path):
    os.makedirs(extract_path)


# Open the zip archive for reading
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Extract all files to the extract directory
    zip_ref.extractall(extract_path)
'''