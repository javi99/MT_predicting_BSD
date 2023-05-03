import pandas as pd
import re
import requests
import os, stat
import time
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager #details: https://pypi.org/project/webdriver-manager/
from dateutil import rrule
from datetime import datetime
import zipfile
import rarfile
import shutil


#data_dl_path = os.path.join(os.path.dirname(__file__), 'storage')
data_dl_path = os.getcwd()+'/data/storage/'


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
prefs = {"download.default_directory" : data_dl_path[:-1]}
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
               time.sleep(10)

time.sleep(45)

driver.close()


# move file name to dl file name match to into for loop?

print(hrefs)
print(titles)
current_time = datetime.now().time()

pd.DataFrame({'title':titles, 'links':hrefs}).to_csv(f'{data_dl_path}download_{current_time}.csv')

files_dl = os.listdir(data_dl_path)

def ziporrar(file):
    if '.zip' in file[-4:]:
        return 'zip'
    elif '.rar' in file[-4:]:
        return 'rar'
    else:
        print(file)

def unzipper(file):
    ftype = ziporrar(file)
    if ftype == 'zip':
        with zipfile.ZipFile(data_dl_path+ file, 'r') as zip_ref:
            zip_ref.extractall(data_dl_path + file[:-4])
        os.remove(data_dl_path + file)

    elif ftype == 'rar':
        try:
            rar = rarfile.RarFile(data_dl_path + file)
            rar.extractall(data_dl_path)
            os.remove(data_dl_path + file)
        except:
            print(file)

        
for file in files_dl:
    unzipper(file)


files_dl = os.listdir(data_dl_path)

for file in files_dl:
    if os.path.isdir(data_dl_path + file):
        datas = os.listdir(data_dl_path + file)
        for data in datas:
            os.chmod(data_dl_path + file + '/' + data, 0o777)
            if '__MACOSX' in data:
                os.rmdir(data_dl_path + file + '/' + data)
                continue
            shutil.move(data_dl_path + file + '/' + data ,data_dl_path)
        os.rmdir(data_dl_path + file)
