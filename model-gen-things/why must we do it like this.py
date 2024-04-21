import os
import requests
import time

status_url = "http://3.145.59.0:8000/api/check_status/"
download_url = "http://3.145.59.0:8000/api/download_ply/"

while(True):
    server_resp = requests.get(status_url)
    print(server_resp.text)
    time.sleep(5)

