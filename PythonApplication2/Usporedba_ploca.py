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

# Example usage
file_paths = [
    ('Outfiles/OutFiles2/CameraPoses.txt', 'Outfiles/OutFiles2/RobotPoses.txt'),
    ('OutFiles/OutFiles2/CameraPosesOpenCV.txt', 'OutFiles/OutFiles2/RobotPosesOpenCV.txt')
]

method_names = ['TSAI', 'ANDREFF', 'DANIILIDIS', 'HORAUD', 'PARK']
error_matrix = {method: [] for method in method_names}

for file_path_cam, file_path_rob in file_paths:
    matrices_cam_full_b = read_matrix_from_file(file_path_cam)
    matrices_rob_full_b = read_matrix_from_file(file_path_rob)
    print(len(matrices_cam_full_b))
    assert len(matrices_cam_full_b) == len(matrices_rob_full_b), "The number of matrices in both files must be the same."

    matrices_cam_full = matrices_cam_full_b
    matrices_rob_full = matrices_rob_full_b

    subset_sizes = range(6, 22, 5)
    methods = [cv2.CALIB_HAND_EYE_TSAI, cv2.CALIB_HAND_EYE_ANDREFF, cv2.CALIB_HAND_EYE_DANIILIDIS, cv2.CALIB_HAND_EYE_HORAUD, cv2.CALIB_HAND_EYE_PARK]

    for method, method_name in zip(methods, method_names):
        error_vector = []
        for i in subset_sizes:
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
            
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base_vec, t_gripper2base_vec, R_target2cam_vec, t_target2cam_vec, method=method)

            cam2gripper = np.eye(4)
            cam2gripper[:3, :3] = R_cam2gripper
            cam2gripper[:3, 3] = t_cam2gripper.flatten()
            # za rotaciju
            # Matrix_A = calculate_transposes(R_gripper2base_vec)
            # Matrix_B = calculate_transposesB(R_target2cam_vec)
            # za translaciju
            Matrix_A = calculate_transposes(matrices_rob)
            Matrix_B = calculate_transposesB(matrices_cam)

            # error = calculate_ER(R_cam2gripper, Matrix_B, Matrix_A) * (180/np.pi)
            error = calculate_ET(cam2gripper, Matrix_B, Matrix_A) * 1000
            error_vector.append(error)
        
        error_matrix[method_name].append(error_vector)

print("Error matrix:", error_matrix)

# Plotting the results
labels = ['ChArUco', 'Sahovnica']
subset_sizes = list(range(6, 22, 5))

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'font.size': 14
})

plt.figure(figsize=(10, 6))
linestyles = ['-', '--']
for method_name, errors in error_matrix.items():
    if errors:
        for idx, error_list in enumerate(errors):
            if error_list:
                plt.plot(subset_sizes, error_list, linestyle=linestyles[idx], label=f"{method_name} {labels[idx]}")

plt.xlabel('Broj polozaja', fontsize=20)
plt.ylabel(r'Pogreska $E_T$ [mm]', fontsize=20)
plt.title('Usporedba kalibracijskih ploca', fontsize = 20)
plt.legend()
plt.grid(True)
plt.show()
