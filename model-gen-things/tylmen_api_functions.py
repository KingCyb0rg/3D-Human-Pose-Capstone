import time
import requests
import open3d as o3d
import numpy as np

def api_master_pipe(mov, interval=10, hush=False):
    api_upload_mov(mov)
    api_generate_ply()
    out = api_wait_and_download(interval)
    return out

def api_upload_mov(mov):
    upload_url = "http://3.145.59.0:8000/api/upload/"   # POST [ to upload .mov file ]
    #TODO

def api_generate_ply():
    generate_url = "http://3.145.59.0:8000/api/generate_ply/"   # GET [to generate .ply file ]
    gen_resp = requests.get(generate_url)

def api_get_sane_status():
    status_url = "http://3.145.59.0:8000/api/check_status/"   # GET [to check if generation process has finished ]
    status_resp = requests.get(status_url)
    if(status_resp.text() == "Genrating ply is done. You may download ply"):
        return 2
    return 0
    
def api_wait_and_download(interval=10):
    download_url = "http://3.145.59.0:8000/api/download_ply/"   # GET [to dowload .ply file ]
    isdone = False
    while(isdone == False):
        api_response = api_get_sane_status()
        if(api_response == 2):
            isdone = True
        if(api_response == 1):
            return -1
        time.sleep(interval)
    down_resp = requests.get(download_url)
    return down_resp