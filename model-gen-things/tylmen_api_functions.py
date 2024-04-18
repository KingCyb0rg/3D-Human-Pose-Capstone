import time
import requests
import open3d as o3d
import numpy as np

#I considered adding handling for http response codes as error checking, but didn't get around to it.


#Main pipe function: Takes a .mov file's binary data and passes it through the entire pipeline. Hush disables
#console output, interval determines how long the app waits between server pings while waiting for the ply
def api_master_pipe(mov, interval=10, hush=False):
    api_upload_mov(mov)
    api_generate_ply()
    out = api_wait_and_download(interval)
    return out.content

#Uploads the file's binary data, which is allegedly the correct way to POST a file.
def api_upload_mov(mov, hush=False):
    upload_url = "http://3.145.59.0:8000/api/upload/"   # POST [ to upload .mov file ]
    files = {'files': mov}
    up_response = requests.post(upload_url, files)

#tells the server to begin generating the cloud.
def api_generate_ply(hush=False):
    generate_url = "http://3.145.59.0:8000/api/generate_ply/"   # GET [to generate .ply file ]
    gen_resp = requests.get(generate_url) #stored for umimplemented error handling reasons

#checks the server's status, using a simple textmatch for the completion state.
def api_get_sane_status(url):
    status_resp = requests.get(url)
    if(status_resp.text() == "Genrating ply is done. You may download ply"):
        return 2
    return 0
    
#begins pinging the server every interval seconds to check if it's done making the cloud. when it returns the completion state message,
#downloads the ply.
def api_wait_and_download(interval=10, hush=False):
    status_url = "http://3.145.59.0:8000/api/check_status/"   # GET [to check if generation process has finished ]
    download_url = "http://3.145.59.0:8000/api/download_ply/"   # GET [to dowload .ply file ]
    isdone = False
    while(isdone == False):
        api_response = api_get_sane_status(status_url)
        if(api_response == 2):
            isdone = True
        if(api_response == 1): #leftover from unimplemented error handling
            return -1
        time.sleep(interval)
    down_resp = requests.get(download_url)
    return down_resp