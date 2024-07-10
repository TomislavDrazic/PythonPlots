
import ast
from re import S
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
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

def calculate_ET(X, B_list, A_list):
    N = len(B_list)
    assert N == len(A_list), "The number of rotation matrices in B_list and A_list must be the same"
    
    t_X = X[:3, 3]
    R_X = X[:3, :3]
    total_result = 0
    
    for B, A in zip(B_list, A_list):
        R_A = A[:3, :3]
        t_A = A[:3, 3]
        t_B = B[:3, 3]
        
        term1 = np.dot(R_A, t_X) - t_X - np.dot(R_X, t_B) + t_A  # Corrected vector operations
        result = np.linalg.norm(term1)  # Ensure the norm of the vector is computed correctly
        total_result += result
        
    E_T = total_result / N
    return E_T

def calculate_transposes(matrices):
    transposes = []
    for i in range(len(matrices)):
        for j in range(i + 1, len(matrices)):
            T_ij = np.dot(np.linalg.inv(matrices[i]), matrices[j])
            transposes.append(T_ij)
    return transposes

def calculate_transposesB(matrices):
    transposes = []
    for i in range(len(matrices)):
        for j in range(i + 1, len(matrices)):
            T_ij = np.dot(matrices[i], np.linalg.inv(matrices[j]))
            transposes.append(T_ij)
    return transposes

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

def calculate_error(A_list, B_list, X):
    N = len(A_list)
    error_sum = 0.0

    for i in range(N):
        A_i = A_list[i]
        B_i = B_list[i]
        AX = np.dot(A_i, X)
        XB = np.dot(X, B_i)
        difference = AX - XB
        norm_squared = np.linalg.norm(difference, 'fro')**2
        error_sum += norm_squared

    error = np.sqrt(error_sum / N)
    return error

# Example usage
file_paths = [
    ('OutFiles/OutFiles2/CameraPoses_t_112.5.txt', 'OutFiles/OutFiles2/RobotPoses_t_112.5.txt'),
    ('OutFiles/OutFiles2/CameraPoses_t_75.txt', 'OutFiles/OutFiles2/RobotPoses_t_75.txt'),
      ('OutFiles/OutFiles2/CameraPoses_t_37.5.txt', 'OutFiles/OutFiles2/RobotPoses_t_37.5.txt'),
    ('OutFiles/OutFiles2/CameraPoses.txt', 'OutFiles/OutFiles2/RobotPoses.txt'),
]

# file_paths = [
#     ('OutFiles/OutFiles2/CameraPoses_r_10.txt', 'OutFiles/OutFiles2/RobotPoses_r_10.txt'),
#     ('OutFiles/OutFiles2/CameraPoses.txt', 'OutFiles/OutFiles2/RobotPoses.txt'),
#     ('OutFiles/OutFiles2/CameraPoses_r_17.5.txt', 'OutFiles/OutFiles2/RobotPoses_r_17.5.txt'),
#     ('OutFiles/OutFiles2/CameraPoses_r_20.txt', 'OutFiles/OutFiles2/RobotPoses_r_20.txt'),
#     ('OutFiles/OutFiles2/CameraPoses_r_30.txt', 'OutFiles/OutFiles2/RobotPoses_r_30.txt'),
# ]

method_names = ['HOURAD','PARK','TSAI','ANDREFF','DANIILIDIS']
error_matrix = {method: [] for method in method_names}

for file_path_cam, file_path_rob in file_paths:
    matrices_cam_full = read_matrix_from_file(file_path_cam)
    matrices_rob_full = read_matrix_from_file(file_path_rob)
    print(len(matrices_cam_full))
    assert len(matrices_cam_full) == len(matrices_rob_full), "The number of matrices in both files must be the same."

    subset_sizes = range(5, 25, 5)
    methods = [cv2.CALIB_HAND_EYE_HORAUD, cv2.CALIB_HAND_EYE_PARK, cv2.CALIB_HAND_EYE_TSAI,cv2.CALIB_HAND_EYE_ANDREFF, cv2.CALIB_HAND_EYE_DANIILIDIS]

    for method, method_name in zip(methods, method_names):
        error_vector = []
        matrices_cam = matrices_cam_full
        matrices_rob = matrices_rob_full

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
            
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base_vec, t_gripper2base_vec, R_target2cam_vec, t_target2cam_vec, method=method)

        cam2gripper = np.eye(4)
        cam2gripper[:3, :3] = R_cam2gripper
        cam2gripper[:3, 3] = t_cam2gripper.flatten()
        # za rotaciju
        Matrix_A = calculate_transposes(R_gripper2base_vec)
        Matrix_B = calculate_transposesB(R_target2cam_vec)
        # za translaciju
        # Matrix_A = calculate_transposes(matrices_rob)
        # Matrix_B = calculate_transposesB(matrices_cam)

        error = calculate_ER(R_cam2gripper, Matrix_B, Matrix_A) *(180/np.pi)
        # error = calculate_ET(cam2gripper, Matrix_B, Matrix_A) * 1000
        # error = calculate_error(Matrix_A, Matrix_B, cam2gripper)
        error_vector.append(error)
        
        error_matrix[method_name].append(error_vector)


# Extract error values and format them for plotting
error_data = []
for method_name, errors in error_matrix.items():
    for idx, error_list in enumerate(errors):
        error_data.append({
            'Method': method_name,
            'File Path Index': idx + 1,  # Using 1 for first file path, 2 for second
            'Error': error_list[0]
        })

# Create a DataFrame for easier plotting
import pandas as pd
error_df = pd.DataFrame(error_data)

# Plotting the results with file path indices on x-axis and errors on y-axis
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'font.size': 20
})
plt.figure(figsize=(12, 6))
for method_name in error_df['Method'].unique():
    subset = error_df[error_df['Method'] == method_name]
    if not subset.empty:
        plt.plot(subset['File Path Index'], subset['Error'], label=f"{method_name}")

plt.xlabel('Raspon translacija')
plt.ylabel(r'Pogreska $E_R$ [deg]')
plt.title('Utjecaj raspona translacija')
plt.legend()
plt.grid(True)
plt.xticks(ticks=[1, 2, 3 ,4], labels=['37,5','75','112,5','150'], rotation=0)
plt.tight_layout()
plt.show()