import yaml
import numpy as np
import cv2

# Load YAML data from file
variable = 1
num_matrices = 8  
Hg = np.empty((num_matrices, 4, 4))
Hc = np.empty((num_matrices, 4, 4))
for i in range(8):
    file_path = rf'C:\Users\38599\Desktop\Faks\Diplomski\Python_folder\Robo_pos\pose_fPe_{variable}.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    variable += 1

    data_trans = data['data'][:3]
    trans = np.array(data_trans)
    data_rot = data['data'][-3:]
    array_data_1 = np.array(data_rot)
    rot = cv2.Rodrigues(array_data_1)[0]
    Mat_rot = np.zeros((4, 4))
    Mat_rot[3, 3] = 1
    Mat_rot[0:3, 0:3] = rot
    Mat_rot[0:3, 3] = trans.flatten()

    Hg[i] = Mat_rot


variable = 1
for i in range(8):
    file_path = rf'C:\Users\38599\Desktop\Faks\Diplomski\Python_folder\Cam_pos\pose_cPo_{variable}.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    variable += 1

    data_trans = data['data'][:3]
    trans = np.array(data_trans)
    data_rot = data['data'][-3:]
    array_data_1 = np.array(data_rot)
    rot = cv2.Rodrigues(array_data_1)[0]
    Mat_rot = np.zeros((4, 4))
    Mat_rot[3, 3] = 1
    Mat_rot[0:3, 0:3] = rot
    Mat_rot[0:3, 3] = trans.flatten()
    Hc[i] = Mat_rot
    print(Mat_rot)
