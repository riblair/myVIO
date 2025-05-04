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
import Path
from PathGrapher import PathGrapher
import utils as util
import ImuUtils
import argparse
import os
from Path import *
from PathGrapher import PathGrapher
from utils import IMU_DT, euler_from_quat
from ImuUtils import *

def env_setup():
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--Outputs", default="Code/Phase2/Data", type=str, help="Parent Directory for data files. Default:'Code/Phase2/Data/'")
    Parser.add_argument("--Path", default="straight_line", type=str, help="Which path object to use. ['straight_line', 'circle', 'sinusoid', 'figure_eight', 'hyperbolic_paraboloid']")
    Args = Parser.parse_args()

    os.makedirs(Args.Outputs, exist_ok=True)
    os.makedirs(Args.Outputs+"Images/", exist_ok=True)
    return Args

def generate_random_path():
    path_int = random.randint(1, 5)
    match path_int:
        case 1:  # Straight Line
            point_min = 10
            point_max = -10
            path = StraightLine(
                point_a=np.array([random.uniform(point_min, point_max), random.uniform(point_min, point_max), random.uniform(point_min, point_max)]),
                point_b=np.array([random.uniform(point_min, point_max), random.uniform(point_min, point_max), random.uniform(point_min, point_max)])
            )
        case 2:  # Circle
            center_min = 10
            center_max = -10
            diameter_min = -10
            diameter_max = 10
            path = Circle(
                center=np.array([random.uniform(center_min, center_max), random.uniform(center_min, center_max), random.uniform(center_min, center_max)]),
                diameter=random.uniform(diameter_min, diameter_max)
            )
        case 3:  # Sinusoid
            point_min = 10
            point_max = -10
            param_max = 10
            param_min = -10
            path = Sinusoid(
                point_a=np.array([random.uniform(point_min, point_max), random.uniform(point_min, point_max), random.uniform(point_min, point_max)]),
                point_b=np.array([random.uniform(point_min, point_max), random.uniform(point_min, point_max), random.uniform(point_min, point_max)]),
                x_params=(random.uniform(param_min, param_max), random.uniform(param_min, param_max), random.uniform(param_min, param_max)),
                y_params=(random.uniform(param_min, param_max), random.uniform(param_min, param_max), random.uniform(param_min, param_max)),
                z_params=(random.uniform(param_min, param_max), random.uniform(param_min, param_max), random.uniform(param_min, param_max)),
            )
        case 4:  # Figure Eight
            x_min = 10
            x_max = -10
            axis_min = 10
            axis_max = -10
            path = FigureEight(
                x_i=np.array([random.uniform(x_min, x_max), random.uniform(x_min, x_max), random.uniform(x_min, x_max)]),
                major_axis=random.uniform(axis_min, axis_max),
                minor_axis=random.uniform(axis_min, axis_max)
            )
        case 5:  # Hyperbolic Paraboloid
            x_min = 10
            x_max = -10
            width_min = 10
            width_max = -10
            path = HyperbolicParaboloid(
                x_i=np.array([random.uniform(x_min, x_max), random.uniform(x_min, x_max), random.uniform(x_min, x_max)]),
                x_width=random.uniform(width_min, width_max),
                y_width=random.uniform(width_min, width_max),
                z_width=random.uniform(width_min, width_max)
            )
    return path
    

def generate_ground_truth(path: Path, file_idx:int, mode='train'):
    times = np.linspace(0, path.t_f, num=int(path.t_f/IMU_DT)) 
    if mode == 'train':
        filepath = f'Code/Phase2/groundtruth/train/traj_{file_idx}.csv'
    elif mode == 'val':
        filepath = f'Code/Phase2/groundtruth/val/traj_{file_idx}.csv'
    elif mode == 'test':
        filepath = f'Code/Phase2/groundtruth/test/traj_{file_idx}.csv'
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        for t in times:
            position = path.get_position(t)  # x, y, z
            orientation = path.get_orientation(t)  # qx, qy, qz, qw

            # Can add images to this
            row_data = [position[0], position[1], position[2],
                        orientation[0], orientation[1], orientation[2], orientation[3]]
            writer.writerow(row_data)


# def generate_training_data(path: Path):
#     times = np.linspace(0, path.t_f, num=int(path.t_f/util.IMU_DT)) 
#     filepath = f'Code/Phase2/Data/traj_{path.name}.csv'
def generate_data(path: Path, file_idx:int, mode='train'):
    times = np.linspace(0, path.t_f, num=int(path.t_f/IMU_DT)) 
    if mode == 'train':
        filepath = f'Code/Phase2/data/train/traj_{file_idx}.csv'
    elif mode == 'val':
        filepath = f'Code/Phase2/data/val/traj_{file_idx}.csv'
    elif mode == 'test':
        filepath = f'Code/Phase2/data/test/traj_{file_idx}.csv'
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        pg = PathGrapher(path)
        times, positions, orientations = pg._generate_ground_truth('euler')
        accel_data = ImuUtils.cal_linear_acc(positions[:, 0], positions[:, 1], positions[:, 2], imu_rate=1.0/util.IMU_DT)
        gyro_data = ImuUtils.cal_angular_vel(orientations[:, 0], orientations[:, 1], orientations[:, 2], imu_rate=1.0/util.IMU_DT)

        acc_err = accel_high_accuracy

        # sets random vibration to accel with RMS for x/y/z axis - 1/2/3 m/s^2, can be zero or changed to other values
        fs = 200
        num_samples = accel_data.shape[0]
        ref = np.zeros((num_samples, 3))
        env = '[0.03 0.001 0.01]-random'
        vib_def = vib_from_env(env, fs)

        real_acc = acc_gen(fs, ref, acc_err, vib_def)
        accel_data = accel_data + real_acc
        accel_data = np.vstack([np.zeros((2,3)), accel_data])
        
        gyro_err = accel_high_accuracy

        # sets sinusoidal vibration to gyro with frequency being 0.5 Hz and amp for x/y/z axis being 6/5/4 deg/s
        env = '[6 5 4]d-0.5Hz-sinusoidal'
        num_samples = gyro_data.shape[0]
        ref = np.zeros((num_samples, 3))
        vib_def = vib_from_env(env, fs)

        real_gyro = acc_gen(fs, ref, gyro_err, vib_def)
        gyro_data = gyro_data + real_gyro
        gyro_data = np.vstack([np.zeros((1,3)), gyro_data])

        im_names = [f"Code/Phase2/Data/Images/traj_{path.name}_{(i):05}.png" for i in range(len(times))]

        data = np.vstack((accel_data[:, 0], accel_data[:, 1], accel_data[:, 2], gyro_data[:, 0], gyro_data[:, 1], gyro_data[:, 2], im_names[:]))
        writer.writerows(data.T)


def generate_everything(num_total_files: int):
    num_train_files = int(num_total_files * 0.6)
    num_val_files = int(num_total_files * 0.2)
    num_test_files = int(num_total_files * 0.2)
    for i in range(num_train_files):
        path = generate_random_path()
        mode = 'train'
        generate_data(path, i, mode=mode)
        generate_ground_truth(path, i, mode=mode)
        hyperparam_filepath = f'Code/Phase2/path_hyperparameters/{mode}/traj_{i}.yaml'
        path.export_hyperparameters(hyperparam_filepath)

    for i in range(num_val_files):
        path = generate_random_path()
        mode = 'val'
        generate_data(path, i, mode=mode)
        generate_ground_truth(path, i, mode=mode)
        hyperparam_filepath = f'Code/Phase2/path_hyperparameters/{mode}/traj_{i}.yaml'
        path.export_hyperparameters(hyperparam_filepath)

    for i in range(num_test_files):
        path = generate_random_path()
        mode = 'test'
        generate_data(path, i, mode=mode)
        generate_ground_truth(path, i, mode=mode)
        hyperparam_filepath = f'Code/Phase2/path_hyperparameters/{mode}/traj_{i}.yaml'
        path.export_hyperparameters(hyperparam_filepath)
    return

if __name__ == '__main__':
    generate_everything(num_total_files=20)

# generate_ground_truth(Circle(center=np.array([0,0,0]), diameter=10.0))
# generate_data(Circle(center = np.array([0, 0, 0]), diameter=10.0), mode='train')
