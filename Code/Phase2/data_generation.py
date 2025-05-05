"""data_generation pipeline

* Choose Path from Path.py
* Given time, find set of positions, orientations, from path methods
* feed in position diffs to ImuUtils to generate accelerometer data
* feed in orinetations? to ImuUtils to generate gyro data
* Add noise to both by their respective "gen" methods
* Use pose+quat as GT data
* Use accel+gyro+[standard image name] to generate data file
* leave room for taking a photo w/ blender stuff (riley)
*  thxs :)

"""

import csv
import random
import numpy as np
from Path import *
from PathGrapher import PathGrapher
import utils as util
import ImuUtils
import argparse
import os
from image_data_generation import generate_image_data

def env_setup():
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--Outputs", default="Code/Phase2/Data/", type=str, help="Parent Directory for data files. Default:'Code/Phase2/Data/'")
    Parser.add_argument("--Path", default="Circle", type=str, help="Which path object to use. ['Straight_Line', 'Circle', 'Sinusoid', 'Figure_Eight', 'Hyperbolic_Paraboloid']")
    Args = Parser.parse_args()

    if Args.Path not in ACCEPTABLE_PATHS:
        print(f"[ERROR] Wrong type given for '--Path' param. Expected one of {ACCEPTABLE_PATHS}, given {Args.Path}")
        exit(1)

    final_out = Args.Outputs+Args.Path+"/"
    os.makedirs(final_out,                  exist_ok=True)
    os.makedirs(final_out+"Train/",         exist_ok=True)
    os.makedirs(final_out+"Train/Images/",  exist_ok=True)
    os.makedirs(final_out+"Test/",          exist_ok=True)
    os.makedirs(final_out+"Test/Images/",   exist_ok=True)
    os.makedirs(final_out+"Val/",           exist_ok=True)
    os.makedirs(final_out+"Val/Images/",    exist_ok=True)
    return Args

# def generate_random_path():
#     path_int = random.randint(1, 5)
#     match path_int:
#         case 1:  # Straight Line
#             point_min = 10
#             point_max = -10
#             path = StraightLine(
#                 point_a=np.array([random.uniform(point_min, point_max), random.uniform(point_min, point_max), random.uniform(point_min, point_max)]),
#                 point_b=np.array([random.uniform(point_min, point_max), random.uniform(point_min, point_max), random.uniform(point_min, point_max)])
#             )
#         case 2:  # Circle
#             center_min = 10
#             center_max = -10
#             diameter_min = -10
#             diameter_max = 10
#             path = Circle(
#                 center=np.array([random.uniform(center_min, center_max), random.uniform(center_min, center_max), random.uniform(center_min, center_max)]),
#                 diameter=random.uniform(diameter_min, diameter_max)
#             )
#         case 3:  # Sinusoid
#             point_min = 10
#             point_max = -10
#             param_max = 10
#             param_min = -10
#             path = Sinusoid(
#                 point_a=np.array([random.uniform(point_min, point_max), random.uniform(point_min, point_max), random.uniform(point_min, point_max)]),
#                 point_b=np.array([random.uniform(point_min, point_max), random.uniform(point_min, point_max), random.uniform(point_min, point_max)]),
#                 x_params=(random.uniform(param_min, param_max), random.uniform(param_min, param_max), random.uniform(param_min, param_max)),
#                 y_params=(random.uniform(param_min, param_max), random.uniform(param_min, param_max), random.uniform(param_min, param_max)),
#                 z_params=(random.uniform(param_min, param_max), random.uniform(param_min, param_max), random.uniform(param_min, param_max)),
#             )
#         case 4:  # Figure Eight
#             x_min = 10
#             x_max = -10
#             axis_min = 10
#             axis_max = -10
#             path = FigureEight(
#                 x_i=np.array([random.uniform(x_min, x_max), random.uniform(x_min, x_max), random.uniform(x_min, x_max)]),
#                 major_axis=random.uniform(axis_min, axis_max),
#                 minor_axis=random.uniform(axis_min, axis_max)
#             )
#         case 5:  # Hyperbolic Paraboloid
#             x_min = 10
#             x_max = -10
#             width_min = 10
#             width_max = -10
#             path = HyperbolicParaboloid(
#                 x_i=np.array([random.uniform(x_min, x_max), random.uniform(x_min, x_max), random.uniform(x_min, x_max)]),
#                 x_width=random.uniform(width_min, width_max),
#                 y_width=random.uniform(width_min, width_max),
#                 z_width=random.uniform(width_min, width_max)
#             )
#     return path
    
def generate_ground_truth(directory:str, times: np.ndarray, positions: np.ndarray, quaternions: np.ndarray):
    filepath = f'{directory}gt_data.csv'
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        data = np.vstack((positions[:, 0], positions[:, 1], positions[:, 2], quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]))
        writer.writerows(data.T)

def generate_data(directory:str, times: np.ndarray, positions: np.ndarray, euler_angles: np.ndarray):

    filepath = f'{directory}input_data.csv'
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        accel_data_gt = ImuUtils.cal_linear_acc(positions[:, 0], positions[:, 1], positions[:, 2], imu_rate=1.0/util.IMU_DT)
        gyro_data_gt = ImuUtils.cal_angular_vel(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2], imu_rate=1.0/util.IMU_DT)

        real_acc = ImuUtils.acc_gen(util.FREQUENCY, accel_data_gt, util.ERR_ACC, util.VIB_DEF_ACC)
        accel_data = np.vstack([np.zeros((2,3)), real_acc])
        
        real_gyro = ImuUtils.gyro_gen(util.FREQUENCY, gyro_data_gt, util.ERR_GYRO, util.VIB_DEF_GYRO)
        gyro_data = np.vstack([np.zeros((1,3)), real_gyro])

        im_names = [f"{directory}Images/im_{(i):05}.png" for i in range(len(times))]

        data = np.vstack((accel_data[:, 0], accel_data[:, 1], accel_data[:, 2], gyro_data[:, 0], gyro_data[:, 1], gyro_data[:, 2], im_names[:]))
        writer.writerows(data.T)

def generate_everything(args):
    if args.Path == "Straight_Line":
        path_train = STRAIGHT_LINE_TRAIN
        path_test  = STRAIGHT_LINE_TEST
        path_val   = STRAIGHT_LINE_VAL
    elif args.Path == 'Circle':
        path_train = CIRCLE_TRAIN
        path_test  = CIRCLE_TEST
        path_val   = CIRCLE_VAL
    elif args.Path == 'Sinusoid':
        path_train = SINUSOID_TRAIN
        path_test  = SINUSOID_TEST
        path_val   = SINUSOID_VAL
    elif args.Path == 'Figure_Eight':
        path_train = FIGURE_EIGHT_TRAIN
        path_test  = FIGURE_EIGHT_TEST
        path_val   = FIGURE_EIGHT_VAL
    elif args.Path == 'Hyperbolic_Paraboloid':
        path_train = HYPERBOLIC_PARABOLOID_TRAIN
        path_test  = HYPERBOLIC_PARABOLOID_TEST
        path_val   = HYPERBOLIC_PARABOLOID_VAL
    else:
        print(f"[ERROR] Wrong type given for '--Path' param. Expected ['straight_line', 'circle', 'sinusoid', 'figure_eight', 'hyperbolic_paraboloid'], given {args.Path}")
        exit(1)
    
    times, positions_train, quaternions_train = PathGrapher(path_train)._generate_ground_truth()
    euler_train = np.array([util.euler_from_quat(orientation) for orientation in quaternions_train])
    dir_train = args.Outputs+args.Path+"/Train/"

    __, positions_test, quaternions_test = PathGrapher(path_test)._generate_ground_truth()
    euler_test = np.array([util.euler_from_quat(orientation) for orientation in quaternions_test])
    dir_test = args.Outputs+args.Path+"/Test/"

    __, positions_val, quaternions_val = PathGrapher(path_val)._generate_ground_truth()
    euler_val = np.array([util.euler_from_quat(orientation) for orientation in quaternions_val])
    dir_val = args.Outputs+args.Path+"/Val/"

    generate_data(dir_train, times, positions_train, euler_train)
    generate_ground_truth(dir_train, times, positions_train, quaternions_train)
    generate_image_data(dir_train, times, positions_train, euler_train)
    hyperparam_filepath = f'{dir_train}metadata.yaml'
    path_train.export_hyperparameters(hyperparam_filepath)

    generate_data(dir_test, times, positions_test, euler_test)
    generate_ground_truth(dir_test, times, positions_test, quaternions_test)
    # generate_image_data(dir_test, times, positions_test, euler_test)
    hyperparam_filepath = f'{dir_test}metadata.yaml'
    path_test.export_hyperparameters(hyperparam_filepath)

    generate_data(dir_val, times, positions_val, euler_val)
    generate_ground_truth(dir_val, times, positions_val, quaternions_val)
    # generate_image_data(dir_val, times, positions_val, euler_val)
    hyperparam_filepath = f'{dir_val}metadata.yaml'
    path_val.export_hyperparameters(hyperparam_filepath)

if __name__ == '__main__':
    args = env_setup()
    generate_everything(args)
