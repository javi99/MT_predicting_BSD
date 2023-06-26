import requests
import os
import time
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager #details: https://pypi.org/project/webdriver-manager/

data_dl_path = "../storage/raw/historical_data/weather_data"

options = webdriver.ChromeOptions()
prefs = {"download.default_directory" : data_dl_path}
options.add_experimental_option("prefs",prefs)
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

driver.get('https://datos.madrid.es/sites/v/index.jsp?vgnextoid=fa8357cec5efa610VgnVCM1000001d4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD')

# click in accept cookies
time.sleep(3)

# Accept cookies
cookies_button = driver.find_element('xpath','//*[@id="iam-cookie-control-modal-action-primary"]')

# Interact with the element by clicking it
cookies_button.click()

# click on "ver mas enlaces" button
see_more_button = driver.find_element("xpath", '//*[@id="readspeaker"]/div[3]/div/div[1]/div/a')
see_more_button.click()

links = driver.find_elements(By.TAG_NAME, 'a')

hrefs = map(lambda link: link.get_attribute("href") if link.get_attribute('class') == "asociada-link ico-csv" else None, links)
clean_hrefs =list(filter(lambda link: link != None , hrefs))

# The hrefs list is inversely ordered by year and month.
# it starts at april 2023, and ends at genuary 2019
hrefs= []
for link in links:
    if link.get_attribute('class') == "asociada-link ico-csv":
        hrefs.append(link.get_attribute('href'))
time.sleep(5)
driver.close()

# download all data and save it in download folder
for href in hrefs:
    r = requests.get(href, allow_redirects=True)
    filename = r.url.split("/")[-1]
    print(filename)
    open(os.path.join(data_dl_path, filename), 'wb').write(r.content)