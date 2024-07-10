import ast
from re import S
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import random

def read_matrix_from_file(file_path):
    matrices = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    matrix_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith('['):
            line = line[1:]
        if line.endswith(']'):
            line = line[:-1]
        if line.endswith(';'):
            line = line[:-1]
        
        if line:
            values = list(map(float, line.split(',')))
            if len(values) != 4:
                raise ValueError("The row does not contain exactly 4 values.")
            matrix_lines.append(values)
        
        if len(matrix_lines) == 4:
            matrices.append(np.array(matrix_lines))
            matrix_lines = []
    
    return matrices


def rotation_matrix_angle(R1, R2):
    """
    Calculate the angle between two rotation matrices.
    """
    R = np.dot(R1.T, R2)
    trace_R = np.trace(R)
    angle = np.arccos((trace_R - 1) / 2)
    return angle

def calculate_ER(R_X, R_B_list, R_A_list):
    N = len(R_B_list)
    assert N == len(R_A_list), "The number of rotation matrices in R_B_list and R_A_list must be the same"
    
    total_angle = 0
    for R_B, R_A in zip(R_B_list, R_A_list):
        transformed_RB = np.dot(R_X, R_B)
        transformed_RA = np.dot(R_A, R_X)
        angle = rotation_matrix_angle(transformed_RB, transformed_RA)
        total_angle += angle
        
    E_R = total_angle / N
    return E_R

def calculate_transposes(matrices):

    transposes = []
    
    for i in range(len(matrices)):
        for j in range(i + 1, len(matrices)):
            # Calculate the transpose (difference) between matrices[i] and matrices[j]
            T_ij = np.dot(np.linalg.inv(matrices[i]), matrices[j])
            transposes.append(T_ij)
    
    return transposes

def calculate_transposesB(matrices):

    transposes = []
    
    for i in range(len(matrices)):
        for j in range(i + 1, len(matrices)):
            # Calculate the transpose (difference) between matrices[i] and matrices[j]
            T_ij = np.dot(matrices[i], np.linalg.inv(matrices[j]))
            transposes.append(T_ij)
    
    return transposes





# Example usage
# file_path_cam = 'OutFiles/CameraPoses.txt'
# file_path_rob = 'OutFiles/RobotPoses.txt'
# file_path_cam = 'Outfiles/CameraPoses_ChArUco.txt'
# file_path_rob = 'Outfiles/RobotPoses_ChArUco.txt'
file_path_cam = 'OutFiles/OutFiles2/CameraPoses.txt'
file_path_rob = 'OutFiles/OutFiles2/RobotPoses.txt'
# file_path_cam = 'Outfiles/CameraPoses_same_height.txt'
# file_path_rob = 'Outfiles/RobotPoses_same_height.txt'
matrices_cam_full_b = read_matrix_from_file(file_path_cam)
matrices_rob_full_b = read_matrix_from_file(file_path_rob)
print(len(matrices_cam_full_b))
print(len(matrices_rob_full_b))

assert len(matrices_cam_full_b) == len(matrices_rob_full_b), "The number of matrices in both files must be the same."

# matrices_cam_full = [matrices_cam_full_b[i] for i in shuffled_indices]
# matrices_rob_full = [matrices_rob_full_b[i] for i in shuffled_indices]
matrices_cam_full = matrices_cam_full_b
matrices_rob_full = matrices_rob_full_b

error_matrix = []
subset_sizes = range(5, 90, 5)
methods = [cv2.CALIB_HAND_EYE_TSAI, cv2.CALIB_HAND_EYE_ANDREFF, cv2.CALIB_HAND_EYE_DANIILIDIS, cv2.CALIB_HAND_EYE_HORAUD, cv2.CALIB_HAND_EYE_PARK]
method_names = ['TSAI', 'ANDREFF', 'DANIILIDIS', 'HORAUD', 'PARK']

for i in subset_sizes:
    error_vector = []
    matrices_cam = matrices_cam_full[:i]
    matrices_rob = matrices_rob_full[:i]

    R_gripper2base_vec = []
    t_gripper2base_vec = []
    R_target2cam_vec = []
    t_target2cam_vec = []

    for matrix in matrices_rob:
        R_gripper2base_vec.append(matrix[:3, :3])
        t_gripper2base_vec.append(matrix[:3, 3])
    
    for matrix in matrices_cam:
        R_target2cam_vec.append(matrix[:3, :3])
        t_target2cam_vec.append(matrix[:3, 3])

    for method in methods:
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base_vec, t_gripper2base_vec, R_target2cam_vec, t_target2cam_vec, method=method)

        cam2gripper = np.eye(4)
        cam2gripper[:3, :3] = R_cam2gripper
        cam2gripper[:3, 3] = t_cam2gripper.flatten()

        Matrix_A = calculate_transposes(R_gripper2base_vec)
        Matrix_B = calculate_transposesB(R_target2cam_vec)

        error = calculate_ER(R_cam2gripper, Matrix_B, Matrix_A) * (180/np.pi)
        error_vector.append(error)
    
    error_matrix.append(error_vector)

# Set the font properties to Times New Roman with size 12
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'font.size': 20
})

# Plotting the results
plt.figure(figsize=(10, 6))

for idx, method_name in enumerate(method_names):
    errors = [error[idx] for error in error_matrix]
    plt.plot(subset_sizes, errors, label=method_name)

plt.xlabel('Broj polozaja')
plt.ylabel(r'Pogreska $E_R$ [deg]')  # LaTeX-style formatting for subscript
plt.title('Kalibracija "oko - u - ruci" za razlicite metode')
plt.legend()
plt.grid(True)
plt.show()