
import os

from extract_measurements import extract_measurements
#from models.smpl import SMPLX_ALL
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import time
import json
import torch
import joblib
import argparse
import numpy as np
from tqdm import tqdm
from multi_person_tracker import MPT
from torch.utils.data import DataLoader
import os.path as osp
from matplotlib.image import imsave
from skimage.transform import resize
from torchvision.transforms import Normalize
#from body_measurements.measurement import Body3D

from core.cfgs import cfg, parse_args_new
from models import hmr, pymaf_net, SMPL
from core import path_config, constants
from datasets.inference import Inference
from utils.renderer import PyRenderer
from utils.imutils import crop
from utils.pose_tracker import run_posetracker
from utils.demo_utils import (
    download_url,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
)

MIN_NUM_FRAMES = 1


def process_image(img_file, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    img = cv2.imread(img_file)[:,:,::-1].copy() # PyTorch does not support negative stride at the moment

    # Assume that the person is centerered in the image
    height = img.shape[0]
    width = img.shape[1]
    center = np.array([width // 2, height // 2])
    scale = max(height, width) / 200

    img_np = crop(img, center, scale, (input_res, input_res))
    img = img_np.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    return img_np, img, norm_img

def run_image_demo(image_path, height):
    #args()
    parse_args_new('configs/pymaf_config.yaml',None)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # ========= Define model ========= #
    model = pymaf_net(path_config.SMPL_MEAN_PARAMS, pretrained=True).to(device)

    # ========= Load pretrained weights ========= #
    model_path='data/pretrained_model/PyMAF_model_checkpoint.pt'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=True)

    # Load SMPL model
    smpl = SMPL(path_config.SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False).to(device)
    model.eval()

    # Preprocess input image and generate predictions
    img_np, img, norm_img = process_image(image_path, input_res=constants.IMG_RES)
    
    with torch.no_grad():
        preds_dict, _ = model(norm_img.to(device))
        output = preds_dict['smpl_out'][-1]
        pred_camera = output['theta'][:, :3]
        pred_vertices = output['verts']

    # Calculate camera parameters for rendering
    camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
    camera_translation = camera_translation[0].cpu().numpy()
    pred_vertices = pred_vertices[0].cpu().numpy()

    ### calculate measures ##
    return extract_measurements(height, pred_vertices)
# def args():
#     parser = argparse.ArgumentParser()
#     # parser.add_argument('--img_file', type=str,
#     #                         help='Path to a single input image')
#     # parser.add_argument('--height', type=float,
#     #                     help='height in cm.')
#     parser.add_argument('--cfg_file', type=str, default='configs/pymaf_config.yaml',
#                         help='config file path.')#help='config file path.',
#     parser.add_argument('--misc', default=None, type=str, nargs="*",#help=argparse.SUPPRESS)
#                         help='other parameters')

#     args = parser.parse_args()
    #parse_args(args)

#if args.img_file is not None:
# m=run_image_demo('/ml-dev/arpit/PyMAF/inputs/female_test_1.jpg', 180)
# print(m)
# if __name__ == '__main__':
#     print()