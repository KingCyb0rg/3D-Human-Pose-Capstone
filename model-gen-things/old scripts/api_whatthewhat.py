import requests
import time

status_url = 'http://3.145.59.0:8000/api/check_status/'
response = requests.get(status_url)
print(response.text)
time.sleep(3)