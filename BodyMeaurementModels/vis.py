import os
from extract_measurements import extract_measurements
import plotly
import plotly.graph_objects as go
import plotly.express as px
import smplx
import cv2
import torch
import numpy as np
from torchvision.transforms import Normalize
from core.cfgs import cfg, parse_args_new
from models import pymaf_net
from core import path_config, constants
from utils.imutils import crop
import os
import torch
import numpy as np
import torch.nn.functional as F
from core import path_config, constants
from utils.imutils import crop

import logging
logger = logging.getLogger(__name__)
measurement = ['height', 'waist', 'belly', 'chest', 'wrist', 'neck', 'arm length', 'thigh', 'shoulder width', 'hips', 'ankle', 'New Chest', 'New Belly']
measurement1 = ['Height', 'Shoulder Width', 'Female waist circumference', 'Male waist circumference', 'Chest circumference', 'Hips circumference', 'Belly circumference', 'Arm length', 'Neck circumference', 'Thigh circumference', 'Wrist circumference', 'Ankle circumference']
measurement2 = {'height' :'Height' , 'shoulder width': 'Shoulder length', 'New Belly':'Female waist circumference',  'waist':'Male waist circumference', 'New Chest':'Chest circumference',  'hips':'Hips circumference', 'belly':'Belly circumference', 'arm length':'Arm length', 'neck':'Neck circumference', 'thigh':'Thigh circumference', 'wrist':'Wrist circumference', 'ankle':'Ankle circumference'}
measurementdict = {'height':'length', 'waist':'circumference', 'belly':'circumference', 'chest':'circumference', 'wrist':'circumference', 'neck':'circumference', 'arm length':'length', 'thigh':'circumference', 'shoulder width':'length', 'hips':'circumference', 'ankle':'circumference', 'New Chest' : 'circumference', 'New Belly' : 'circumference'}
LANDMARK_INDICES = {"HEAD_TOP": 412,
                    "HEAD_LEFT_TEMPLE": 166,
                    "NECK_ADAM_APPLE": 3050,
                    "LEFT_HEEL": 3458,
                    "RIGHT_HEEL": 6858,
                    "LEFT_NIPPLE": 3042,
                    "RIGHT_NIPPLE": 6489,

                    "SHOULDER_TOP": 3068,
                    "INSEAM_POINT": 3149,
                    "BELLY_BUTTON": 3501,
                    "BACK_BELLY_BUTTON": 3022,
                    "CROTCH": 1210,
                    "PUBIC_BONE": 3145,
                    "RIGHT_WRIST": 5559,
                    "LEFT_WRIST": 2241,
                    "RIGHT_BICEP": 4855,
                    "RIGHT_FOREARM": 5197,
                    "LEFT_SHOULDER": 3011,
                    "RIGHT_SHOULDER": 6470,
                    "LEFT_ANKLE": 3334,
                    "LOW_LEFT_HIP": 3134,
                    "LEFT_THIGH": 947,
                    "LEFT_CALF": 1074,
                    "LEFT_ANKLE": 3325
                    }

LANDMARK= ["HEAD_TOP","HEAD_LEFT_TEMPLE","NECK_ADAM_APPLE","LEFT_HEEL","RIGHT_HEEL","LEFT_NIPPLE",
                    "RIGHT_NIPPLE","SHOULDER_TOP","INSEAM_POINT","BELLY_BUTTON","BACK_BELLY_BUTTON",
                    "CROTCH","PUBIC_BONE","RIGHT_WRIST","LEFT_WRIST","RIGHT_BICEP","RIGHT_FOREARM",
                    "LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_ANKLE","LOW_LEFT_HIP","LEFT_THIGH","LEFT_CALF",
                    "LEFT_ANKLE"]


def create_mesh_plot(verts: np.ndarray, faces: np.ndarray):
    '''
    Visualize smpl body mesh.
    :param verts: np.array (N,3) of vertices
    :param faces: np.array (F,3) of faces connecting the vertices

    Return:
    plotly Mesh3d object for plotting
    '''
    mesh_plot = go.Mesh3d(
                        x=verts[:,0],
                        y=verts[:,1],
                        z=verts[:,2],
                        color="gray",
                        hovertemplate ='<i>Index</i>: %{text}',
                        text = [i for i in range(verts.shape[0])],
                        # i, j and k give the vertices of triangles
                        i=faces[:,0],
                        j=faces[:,1],
                        k=faces[:,2],
                        opacity=0.6,
                        name='body',
                        )
    return mesh_plot


def create_wireframe_plot(verts: np.ndarray,faces: np.ndarray):
    '''
    Given vertices and faces, creates a wireframe of plotly segments.
    Used for visualizing the wireframe.
    
    :param verts: np.array (N,3) of vertices
    :param faces: np.array (F,3) of faces connecting the verts
    '''
    i=faces[:,0]
    j=faces[:,1]
    k=faces[:,2]

    triangles = np.vstack((i,j,k)).T

    x=verts[:,0]
    y=verts[:,1]
    z=verts[:,2]

    vertices = np.vstack((x,y,z)).T
    tri_points = vertices[triangles]

    #extract the lists of x, y, z coordinates of the triangle 
    # vertices and connect them by a "line" by adding None
    # this is a plotly convention for plotting segments
    Xe = []
    Ye = []
    Ze = []
    for T in tri_points:
        Xe.extend([T[k%3][0] for k in range(4)]+[ None])
        Ye.extend([T[k%3][1] for k in range(4)]+[ None])
        Ze.extend([T[k%3][2] for k in range(4)]+[ None])

    # return Xe, Ye, Ze 
    wireframe = go.Scatter3d(
                    x=Xe,
                    y=Ye,
                    z=Ze,
                    mode='lines',
                    name='wireframe',
                    line=dict(color= 'rgb(70,70,70)', width=1)
                    )
    return wireframe





def create_measurement_circumference_plot(  cploc, measurementvalues,
                                            measurement_name: str,
                                            verts: np.ndarray,
                                            faces: np.ndarray,
                                            color: str):
    '''
    Create circumference measurement plot
    :param measurement_name: str, measurement name to plot
    :param verts: np.array (N,3) of vertices
    :param faces: np.array (F,3) of faces connecting the vertices
    :param color: str of color to color the measurement

    Return
    plotly object to plot
    '''
    measurement_landmarks_inds = cploc[measurement_name]
    m_viz_name = f"{measurement2[measurement_name]}: {measurementvalues[measurement_name]:.2f}cm"

    segments = {"x":[],"y":[],"z":[]}
    
    for loc in measurement_landmarks_inds:
        segments["x"].append(verts[loc][0])
        segments["y"].append(verts[loc][1])
        segments["z"].append(verts[loc][2])
        
    return go.Scatter3d(
            x=segments["x"],
            y=segments["y"],
            z=segments["z"],
            mode="lines",
            line=dict(
                color=color,
                width=10),
            name=m_viz_name
                )


def create_measurement_length_plot(cploc, measurementvalues,
                                    measurement_name: str,
                                    verts: np.ndarray,
                                    color: str
                                    ):
    '''
    Create length measurement plot.
    :param measurement_name: str, measurement name to plot
    :param verts: np.array (N,3) of vertices
    :param color: str of color to color the measurement

    Return
    plotly object to plot
    '''
    
    measurement_landmarks_inds = cploc[measurement_name]

    segments = {"x":[],"y":[],"z":[]}
    
    for loc in measurement_landmarks_inds:
        segments["x"].append(verts[loc][0])
        segments["y"].append(verts[loc][1])
        segments["z"].append(verts[loc][2])
    for ax in ["x","y","z"]:
        segments[ax].append(None)

    if measurement_name in measurementvalues:
        m_viz_name = f"{measurement2[measurement_name]}: {measurementvalues[measurement_name]:.2f}cm"
    else:
        m_viz_name = measurement_name

    return go.Scatter3d(x=segments["x"], 
                                y=segments["y"], 
                                z=segments["z"],
                                marker=dict(
                                        size=4,
                                        color="rgba(0,0,0,0)",
                                    ),
                                    line=dict(
                                        color=color,
                                        width=10),
                                    name=m_viz_name
                                    )

        

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

    model.eval()

    # Preprocess input image and generate predictions
    img_np, img, norm_img = process_image(image_path, input_res=constants.IMG_RES)
    
    with torch.no_grad():
        preds_dict, _ = model(norm_img.to(device))
        output = preds_dict['smpl_out'][-1]
        pred_camera = output['theta'][:, :3]
        pred_vertices = output['verts']
        kp_3d = output['kp_3d']

    # Calculate camera parameters for rendering
    camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
    camera_translation = camera_translation[0].cpu().numpy()
    pred_vertices = pred_vertices[0].cpu().numpy()
    kp_3d = kp_3d[0].cpu().numpy()
    
    return pred_vertices,kp_3d


def convert_cp():
    
  f = open(os.path.join('data/customBodyPoints.txt'), "r")

  tmplist = []
  cp = []
  for line in f:
    if '#' in line:
      if len(tmplist) != 0:
        cp.append(tmplist)
        tmplist = []
    elif len(line.split()) == 1:
      continue
    else:
      tmplist.append(list(map(float, line.strip().split())))
  cp.append(tmplist)
  return cp

def create_landmarks_plot(kp_3d, color, lm_name
                              ):
    '''
    Visualize landmarks from landmark_names list
    :param landmark_names: List[str] of landmark names to visualize

    Return
    :param plots: list of plotly objects to plot
        '''


    return go.Scatter3d(x = [kp_3d[0]],
                        y = [kp_3d[1]], 
                        z = [kp_3d[2]], 
                        mode='markers',
                        marker=dict(size=8,
                                    color=color,
                                    opacity=1,
                                    ),
                        name=lm_name
                        )


def plotnewchest(verts):
    '''
    Create circumference measurement plot
    :param measurement_name: str, measurement name to plot
    :param verts: np.array (N,3) of vertices
    :param faces: np.array (F,3) of faces connecting the vertices
    :param color: str of color to color the measurement

    Return
    plotly object to plot
    '''
    segments = {"x":[],"y":[],"z":[]}
    chestverts = [3076,2870,1254,1255,1349,1351,3033,3030,3037,3039,611,610,2867,2865,2952,2953,
                  2954, 2955, 763, 761, 2908, 1236, 750, 751, 3014, 4239, 4719, 4380, 4249,
                4251, 4224, 4133, 4132, 4896,4683, 4681, 4897, 4894, 4156, 4086, 4089, 
                4174, 4172, 4737, 6332, 3076]
    chestverts1 = [3076,2870,1254,1255,1349,1351,3033,3030,3037,3039,611,610,2867,2865,2952,2953,
                  2954, 2955, 763, 761, 2908, 1236, 750, 751, 3014, 4239, 4719, 4380, 4249,
                4251, 4224, 4133, 4132, 4896,4098, 6490, 6482, 6478, 4825, 4740, 4737, 6332, 3076]
    bellyverts = [1326, 1327, 922, 923, 3478, 2831, 1484, 1485, 2829, 1488, 1489, 1265, 1266, 2824,
                  1496, 1493, 2827, 3173,6298, 6287, 6288, 4965, 4966, 6285, 4749, 4750, 4961, 4962,
                  6289,4957, 4958, 6291, 6876,4409,4410,4806,1326]
    for loc in chestverts1:
        segments["x"].append(verts[loc][0])
        segments["y"].append(verts[loc][1])
        segments["z"].append(verts[loc][2])
        
    return go.Scatter3d(
            x=segments["x"],
            y=segments["y"],
            z=segments["z"],
            mode="lines",
            line=dict(
                color='red',
                width=10)
                )

def show3Dmesh(verts,measurementvalues, uuid,
                  title="Measurement visualization"):
    '''
    Visualize the SMPL mesh with measurements and landmarks.
    :param title: str, title of plot
    '''
    cp = convert_cp()
    cploc = {}
    for i in range(len(cp)):
        value = []
        for j in range(len(cp[i])):
            value.append(int(cp[i][j][1]))
        cploc[measurement[i]] = value
    
    #faces = smplx.SMPL('data/smpl/SMPL_MALE.pkl').faces
    faces = smplx.SMPL('data/smpl/SMPL_NEUTRAL.pkl').faces
        

    # visualize model mesh
    fig = go.Figure()
    mesh_plot = create_mesh_plot(verts, faces)
    fig.add_trace(mesh_plot)

    # visualize wireframe
    wireframe_plot = create_wireframe_plot(verts, faces)
    fig.add_trace(wireframe_plot)
    
    
    #fig.add_trace(plotnewchest(verts))
    
    #visualize 3D keypoints
    ''' list1 = [str(i) for i in range(0,50)]
    px.colors.qualitative.Alphabet.extend(px.colors.qualitative.Dark24)
    landmark_colors = dict(zip(list1,px.colors.qualitative.Alphabet))
    for i,lm_name in enumerate(kp_3d):
        kp_3d_plot = create_landmarks_plot(kp_3d[i], landmark_colors[str(i)], str(i))
        fig.add_trace(kp_3d_plot) '''

    # visualize measurements
    measurement_colors = dict(zip(measurement,
                                px.colors.qualitative.Alphabet))

    for m_name in measurement:
        if m_name not in measurementdict.keys():
            print(f"Measurement {m_name} not defined.")
            pass
        if m_name == 'chest' or m_name == 'shoulder width':
            continue
        
        if measurementdict[m_name] == 'length':
            
            measurement_plot = create_measurement_length_plot(cploc,measurementvalues, measurement_name=m_name,
                                                                verts=verts,
                                                                color=measurement_colors[m_name])
            
            
        elif measurementdict[m_name] == 'circumference':
            
            measurement_plot = create_measurement_circumference_plot(cploc,measurementvalues, measurement_name=m_name,
                                                                        verts=verts,
                                                                        faces=faces,
                                                                        color=measurement_colors[m_name])
            
        fig.add_trace(measurement_plot)  
            

    fig.update_layout(scene_aspectmode='data',
                        width=1000, height=700,
                        title=title,
                        )
        
    #fig.show()
    rootdir = '/images/'
    path = rootdir + uuid + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    fig.write_html(path + "vis.html")
    return path
    #return os.getcwd() + '/file.html'

def read_html_file(file_path):
    with open(file_path, 'r') as file:
        html_content = file.read()
    return html_content      
  
def show(img, height, uuid):
    pred_vertices , kp_3d = run_image_demo(img, height)
    measurementvalues = extract_measurements(height, pred_vertices)
    html_file_path = show3Dmesh(pred_vertices, measurementvalues, uuid)
    return html_file_path
    #html_content = read_html_file(html_file_path)
    #return html_content
    



def main():
    height = 165.1 #186.0
    #img = '/home/shubham/code/BodyMeaurementModels/inputs/Lloyd/IMG_0477.jpg' #186.0
    #img = '/home/shubham/code/BodyMeaurementModels/inputs/IMG_0524.PNG' #height 165.1 cm
    #img = '/home/shubham/code/BodyMeaurementModels/inputs/IMG_0627.jpg' #height 167 cm
    img = '/home/shubham/code/BodyMeaurementModels/inputs/IMG_0632.jpg' #167 cm
    img = '/home/shubham/code/BodyMeaurementModels/inputs/Lloyd/image0_720.png' #177
    img = '/home/shubham/code/BodyMeaurementModels/inputs/Lloyd/IMG_0679.jpg'
    #img = '/home/shubham/code/BodyMeaurementModels/inputs/views/1556440-500w.jpg'
    #img = '/home/shubham/code/BodyMeaurementModels/inputs/views/Screenshot 2023-06-27 at 3.18.40 PM.png'
    img = '/home/shubham/code/BodyMeaurementModels/inputs/others/IMG_8846.jpeg' #165.1
    img = '/home/shubham/code/BodyMeaurementModels/inputs/views/Screenshot 2023-06-27 at 3.18.40 PM.png'
    #pred_vertices , pred_camera, img_np = run_image_demo(img, 165.1)
    pred_vertices , kp_3d = run_image_demo(img, height)
    measurementvalues = extract_measurements(height, pred_vertices)
    path = show3Dmesh(pred_vertices, measurementvalues)
    a = 1
    
    


if __name__ == '__main__':
    main()
