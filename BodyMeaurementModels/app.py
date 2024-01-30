#!/sizing_api/PyMAF/pymaf_env/bin/python

#API STUFF
from fastapi import FastAPI, File, Form, UploadFile
#import aiofiles
import os
import shutil

#COMPUTER VISION STUFF
#import cv2
#from PIL import Image
#import numpy as np
#from demo import main
#from inference import DeepLabModel, create_pascal_label_colormap, label_to_color_image, LABEL_NAMES, FULL_LABEL_MAP, FULL_COLOR_MAP, MODEL_NAME, _DOWNLOAD_URL_PREFIX, _MODEL_URLS
#from hbmucv.inference import DeepLabModel, create_pascal_label_colormap, label_to_color_image
#import importlib.util
#sp = importlib.util.spec_from_file_location("utils","/sizing_api/Human-Body-Measurements-using-Computer-Vision/utils.py")
#utils = importlib.util.module_from_spec(sp)
#sp.loader.exec_module(utils)

#spec = importlib.util.spec_from_file_location("extract_measurements","/sizing_api/Human-Body-Measurements-using-Computer-Vision/extract_measurements.py")
#extract_measurements = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(extract_measurements)

#spec1 = importlib.util.spec_from_file_location("demo","/sizing_api/Human-Body-Measurements-using-Computer-Vision/demo.py")
#demo = importlib.util.module_from_spec(spec1)
#spec1.loader.exec_module(demo)

#spec2 = importlib.util.spec_from_file_location("inference","/sizing_api/Human-Body-Measurements-using-Computer-Vision/inference.py")
#inference = importlib.util.module_from_spec(spec2)
#spec2.loader.exec_module(inference)



#SIZING INFERENCE
#pics up on line 147 of inference.py: https://github.com/farazBhatti/Human-Body-Measurements-using-Computer-Vision/blob/master/inference.py
"""
_TARBALL_NAME = _MODEL_URLS[MODEL_NAME]
model_dir = 'deeplab_model'
if not os.path.exists(model_dir):
    tf.gfile.MakeDirs(model_dir)

download_path = os.path.join(model_dir, _TARBALL_NAME)
if not os.path.exists(download_path):
    print('downloading model to %s, this might take a while...' % download_path)
    urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME], download_path)
    print('download completed! loading DeepLab model...')

MODEL = DeepLabModel(download_path)
print('model loaded successfully!')
"""

import run_pymaf

"""
def infer(imgg, height):
    _TARBALL_NAME = _MODEL_URLS[MODEL_NAME]
    model_dir = 'deeplab_model'
    if not os.path.exists(model_dir):
        tf.gfile.MakeDirs(model_dir)

    download_path = os.path.join(model_dir, _TARBALL_NAME)
    if not os.path.exists(download_path):
        print('downloading model to %s, this might take a while...' % download_path)
        urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME], download_path)
        print('download completed! loading DeepLab model...')

    MODEL = DeepLabModel(download_path)
    print('model loaded successfully!')
    #dont need this bc the image is already going to be open
    image = Image.open(imgg)
    #image = imgg
    back = cv2.imread('sample_data/input/background.jpeg',cv2.IMREAD_COLOR)

    res_im,seg=MODEL.run(image)

    seg=cv2.resize(seg.astype(np.uint8),image.size)
    mask_sel=(seg==15).astype(np.float32)
    mask = 255*mask_sel.astype(np.uint8)

    img =   np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)   

    res = cv2.bitwise_and(img,img,mask = mask)
    bg_removed = res + (255 - cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

    ans = main(bg_removed, height,None)

    #re-instantiating the model to hopefuly avoid the errors around rerunning the inference code:
    #MODEL = DeepLabModel(download_path)
    #print('model loaded successfully!')
    #tf.reset_default_graph()

    return ans
"""


sizing_api = FastAPI()


@sizing_api.get("/")
async def root():
        return {"message": "Hello World"}

@sizing_api.post("/images_naive/")
async def post_image_naive(file: UploadFile = File(...)):
    with open(file.filename, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
        #testing
        print(file.filename)
    return {"message" : "successfully uploaded %s" % file.filename}

@sizing_api.post("/images/")
async def post_image(file: UploadFile = File(...), apikey: str = Form(), uuid: str = Form(), height: int = Form()):
    handle = ""
    d = {}
    try:
        images_root = "/images/"
        handle = os.path.join(images_root, uuid)
        with open(handle, 'wb') as f:
            while True:
                contents = file.file.read(1024*1024)
                f.write(contents)
                if not contents:
                    break
        file.file.close()
        #d = infer(handle, height)

        d = run_pymaf.run_image_demo(handle, height)


    except Exception as e:
        return {"message": "There was an error uploading the file: %s" % e}
    finally:
        #TESTING
        #d = infer(handle, height)
        #file.file.close()
        if os.path.exists(handle):
            os.remove(handle)

    #print("type:%s " % type(d))
    #print(d)
    
    """
    ddd = {}
    for k, v in d.items():
        ddd[k] = v[0]
    
    #return {"message": "successfully uploaded %s" % file.filename}
    dd = {
        "filename"  : file.filename,
        "apikey"    : apikey,
        "uuid"      : uuid,
        "height"    : height
     }

    return ddd
    """

    return d


#TODO
#add query paramter of height
#add query parameter of UUID

#TODO
#import sizing stuff and test on image
