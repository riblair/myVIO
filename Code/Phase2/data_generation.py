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
import numpy as np
import Path
from PathGrapher import PathGrapher
import utils as util
import ImuUtils
import argparse
import os

def env_setup():
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--Outputs", default="Code/Phase2/Data", type=str, help="Parent Directory for data files. Default:'Code/Phase2/Data/'")
    Parser.add_argument("--Path", default="straight_line", type=str, help="Which path object to use. ['straight_line', 'circle', 'sinusoid', 'figure_eight', 'hyperbolic_paraboloid']")
    Args = Parser.parse_args()

    os.makedirs(Args.Outputs, exist_ok=True)
    os.makedirs(Args.Outputs+"Images/", exist_ok=True)
    return Args

def generate_ground_truth(path: Path):
    times = np.linspace(0, path.t_f, num=int(path.t_f/util.IMU_DT)) 
    filepath = f'Code/Phase2/Data/traj_{path.name}.csv'
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        for t in times:
            position = path.get_position(t)  # x, y, z
            orientation = path.get_orientation(t)  # qx, qy, qz, qw

            # Can add images to this
            row_data = [position[0], position[1], position[2],
                        orientation[0], orientation[1], orientation[2], orientation[3]]
            writer.writerow(row_data)

def generate_training_data(path: Path):
    times = np.linspace(0, path.t_f, num=int(path.t_f/util.IMU_DT)) 
    filepath = f'Code/Phase2/Data/traj_{path.name}.csv'
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        p_position = None
        p_orientation = None
        pp_position = None
        pp_orientation = None
        # TODO: replace this for loop with PathGraphhers _generate_ground_truth() method
        pg = PathGrapher(path)
        times, positions, orientations = pg._generate_ground_truth('euler')
        accel_data = ImuUtils.cal_linear_acc(positions[:, 0], positions[:, 1], positions[:, 2], imu_rate=1.0/util.IMU_DT)
        gyro_data = ImuUtils.cal_angular_vel(orientations[:, 0], orientations[:, 1], orientations[:, 2], imu_rate=1.0/util.IMU_DT)

        accel_data = np.vstack([np.zeros((2,3)), accel_data])
        gyro_data = np.vstack([np.zeros((1,3)), gyro_data])

        im_names = [f"Code/Phase2/Data/Images/traj_{path.name}_{(i):05}.png" for i in range(len(times))]

        data = np.vstack((accel_data[:, 0], accel_data[:, 1], accel_data[:, 2], gyro_data[:, 0], gyro_data[:, 1], gyro_data[:, 2], im_names[:]))
        writer.writerows(data.T)

if __name__ == '__main__':
    args = env_setup()
    if args.Path == "straight_line":
        path = Path.STRAIGHT_LINE
    elif args.Path == 'circle':
        path = Path.CIRCLE
    elif args.Path == 'sinusoid':
        path = Path.SINUSOID
    elif args.Path == 'figure_eight':
        path = Path.FIGURE_EIGHT
    elif args.Path == 'hyperbolic_paraboloid':
        path = Path.HYPERBOLIC_PARABOLOID
    else:
        print(f"[ERROR] Wrong type given for '--Path' param. Expected ['straight_line', 'circle', 'sinusoid', 'figure_eight', 'hyperbolic_paraboloid'], given {args.Path}")
        exit(1)
    
    generate_ground_truth(path)
    generate_training_data(path)