import time
import requests
import open3d as o3d
import numpy as np
#import random
from timeit import default_timer as time_stamp

#I considered adding handling for http response codes and error checking, but didn't get around to it.


#Main pipe function: tells server to start generating
def api_download_pipe(interval=15, hush=False):
    #api_upload_mov(mov) #web app handles this bit
    out = api_wait_and_download(interval)
    return out.content

#Uploads the file's binary data. Unused because the webapp directly uploads the file before running this app.
def api_upload_mov(mov, hush=False):
    upload_url = "http://3.145.59.0:8000/api/upload/"   # POST [ to upload .mov file ]
    files = {'files': mov}
    up_response = requests.post(upload_url, files)
    #print(up_response.text)

#checks the server's status, using a simple textmatch for the completion state.
def api_get_sane_status(url):
    status_resp = requests.get(url)
    print(status_resp.text) #comment this out if you want to declutter the output
    if(status_resp.text == ""):
        return 2
    if(status_resp.text == " "):
        return 2
    return 0
    
#begins pinging the server every interval seconds to check if it's done making the cloud. when it returns the completion state message,
#downloads the ply.


##PENDING REWRITE
def api_wait_and_download(interval=15, hush=False):
    status_url = "http://3.145.59.0:8000/api/check_status/"   # GET [to check if generation process has finished ]
    download_url = "http://3.145.59.0:8000/api/download_ply/"   # GET [to dowload .ply file ]
    isdone = False
    pings = 0
    starttime= time_stamp()
    while(isdone == False):
        time.sleep(interval)
        api_response = api_get_sane_status(status_url)
        if(api_response == 2):
            isdone = True
        else:
            pings += 1
            print("Ping #" + str(pings) + ": not completed")
        #if(api_response == 1): #leftover from unimplemented error checking
    endtime = time_stamp()
    print("PointCloud generation done in " + str(endtime-starttime) + " seconds, downloading")
    down_resp = requests.get(download_url)
    print("Downloaded")
    return down_resp