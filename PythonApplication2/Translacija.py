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
from scipy.linalg import logm
from scipy.linalg import svd




def skew(vector):
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


def tsai(A_list, B_list):
    n = len(A_list)
    S = np.zeros((3 * n, 3))
    v = np.zeros((3 * n, 1))

    # Calculate best rotation R
    for i in range(n):
        A1 = logm(A_list[i][:3, :3])
        B1 = logm(B_list[i][:3, :3])
        a = np.array([A1[2, 1], A1[0, 2], A1[1, 0]]).reshape(-1, 1)
        a = a / np.linalg.norm(a)
        b = np.array([B1[2, 1], B1[0, 2], B1[1, 0]]).reshape(-1, 1)
        b = b / np.linalg.norm(b)
        S[3 * i:3 * i + 3, :] = skew(a.flatten() + b.flatten())
        v[3 * i:3 * i + 3, :] = a - b

    x = np.linalg.lstsq(S, v, rcond=None)[0]
    theta = 2 * np.arctan(np.linalg.norm(x))
    x = x / np.linalg.norm(x)
    R = (np.eye(3) * np.cos(theta) + np.sin(theta) * skew(x.flatten()) + 
         (1 - np.cos(theta)) * np.outer(x, x)).T

    # Calculate best translation t
    C = np.zeros((3 * n, 3))
    d = np.zeros((3 * n, 1))
    I = np.eye(3)
    for i in range(n):
        C[3 * i:3 * i + 3, :] = I - A_list[i][:3, :3]
        d[3 * i:3 * i + 3, :] = A_list[i][:3, 3].reshape(-1, 1) - R @ B_list[i][:3, 3].reshape(-1, 1)

    t = np.linalg.lstsq(C, d, rcond=None)[0]

    # Put everything together to form X
    X = np.eye(4)
    X[:3, :3] = R
    X[:3, 3] = t.flatten()

    return X

def andreff(A_list, B_list):
    n = len(A_list)

    A = np.zeros((12 * n, 12))
    b = np.zeros((12 * n, 1))

    for i in range(n):
        Ra = A_list[i][:3, :3]
        Rb = B_list[i][:3, :3]
        ta = A_list[i][:3, 3]
        tb = B_list[i][:3, 3]

        A[12 * i:12 * i + 9, 0:9] = np.eye(9) - np.kron(Rb, Ra)
        A[12 * i + 9:12 * i + 12, :] = np.hstack([np.kron(tb.reshape(-1, 1).T, np.eye(3)), np.eye(3) - Ra])
        b[12 * i + 9:12 * i + 12, 0] = ta

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    X = x[0:9].reshape(3, 3).T
    X = np.sign(np.linalg.det(X)) / np.abs(np.linalg.det(X))**(1/3) * X

    U, S, Vt = svd(X)
    X = np.dot(U, Vt)
    if np.linalg.det(X) < 0:
        X = np.dot(U, np.diag([1, 1, -1])) @ Vt

    X_final = np.eye(4)
    X_final[:3, :3] = X.T
    X_final[:3, 3] = x[9:12].flatten()

    return X_final



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
            # Calculate the transpose (difference) between matrices[i] and matrices[j]
            T_ij = np.dot(np.linalg.inv(matrices[j]), matrices[i])
            transposes.append(T_ij)
    
    return transposes

def calculate_transposesB(matrices):

    transposes = []
    
    for i in range(len(matrices)):
        for j in range(i + 1, len(matrices)):
            # Calculate the transpose (difference) between matrices[i] and matrices[j]
            T_ij = np.dot(matrices[j], np.linalg.inv(matrices[i]))
            transposes.append(T_ij)
    
    return transposes

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
# file_path_cam = 'Outfiles/Camera_Poses_10.txt'
# file_path_rob = 'Outfiles/Robot_Poses_10.txt'
# file_path_cam = 'OutFiles/CameraPoses_OpenCV.txt'
# file_path_rob = 'OutFiles/RobotPoses_OpenCV.txt'
# file_path_rob = 'Outfiles/OutFiles1/RobotPosesVSP.txt'
# file_path_cam = 'Outfiles/OutFiles1/CameraPosesVSP.txt'
file_path_rob = 'Outfiles/OutFiles2/RobotPoses.txt'
file_path_cam = 'Outfiles/OutFiles2/CameraPoses.txt'
matrices_cam_full = read_matrix_from_file(file_path_cam)
matrices_rob_full = read_matrix_from_file(file_path_rob)
print(len(matrices_cam_full))
print(len(matrices_rob_full))

assert len(matrices_cam_full) == len(matrices_rob_full), "The number of matrices in both files must be the same."


error_matrix = []
subset_sizes = range(5, 90, 5)
methods = [cv2.CALIB_HAND_EYE_TSAI,cv2.CALIB_HAND_EYE_ANDREFF, cv2.CALIB_HAND_EYE_DANIILIDIS, cv2.CALIB_HAND_EYE_HORAUD, cv2.CALIB_HAND_EYE_PARK]
method_names = ['TSAI','ANDREFF', 'DANIILIDIS', 'HORAUD', 'PARK']

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

        Matrix_A = calculate_transposes(matrices_rob)
        Matrix_B = calculate_transposesB(matrices_cam)

        error = calculate_ET(cam2gripper, Matrix_B, Matrix_A) * 1000
        #error = calculate_error(Matrix_A,Matrix_B,cam2gripper)
        error_vector.append(error)
    
    # X = tsai(Matrix_A,Matrix_B)
    # error = calculate_ET(X,Matrix_A,Matrix_B)
    # error_vector.append(error)
    # X = andreff(Matrix_A,Matrix_B)
    # error = calculate_ET(X,Matrix_A,Matrix_B)
    # error_vector.append(error)
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
    #print(errors)
    plt.plot(subset_sizes, errors, label=method_name)

plt.xlabel('Broj polozaja')
plt.ylabel(r'Pogreska $E_T$ [mm]')  # LaTeX-style formatting for subscript
plt.title('Kalibracija "oko - u - ruci" za razlicite metode')
plt.legend()
plt.grid(True)
plt.show()

