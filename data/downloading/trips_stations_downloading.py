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

data_dl_path = os.path.join(os.path.dirname(__file__), 'stations_raw')

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

time.sleep(10)

driver.close()

print(hrefs)
print(titles)
current_time = datetime.now().time()

pd.DataFrame({'title':titles, 'links':hrefs}).to_csv(f'{data_dl_path}/download_{current_time}.csv')

# unzip files
