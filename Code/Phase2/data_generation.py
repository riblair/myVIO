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
from Path import Path, Circle
from PathGrapher import PathGrapher
from utils import IMU_DT, euler_from_quat
import ImuUtils

def generate_ground_truth(path: Path, traj_num: int):
    path = Circle(center = np.array([0, 0, 0]), diameter=10.0)  # Just an example to help write code, will be changed to be general.
    times = np.linspace(0, path.t_f, num=int(path.t_f/IMU_DT)) 
    filepath = f'Code/Phase2/data/traj_{traj_num}.csv'
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
    times = np.linspace(0, path.t_f, num=int(path.t_f/IMU_DT)) 
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
        accel_data = ImuUtils.cal_linear_acc(positions[:, 0], positions[:, 1], positions[:, 2], imu_rate=1.0/IMU_DT)
        gyro_data = ImuUtils.cal_angular_vel(orientations[:, 0], orientations[:, 1], orientations[:, 2], imu_rate=1.0/IMU_DT)

        accel_data = np.vstack([np.zeros((2,3)), accel_data])
        gyro_data = np.vstack([np.zeros((1,3)), gyro_data])

        im_names = [f"Code/Phase2/Data/Images/traj_{path.name}_{(i):05}.png" for i in range(len(times))]

        data = np.vstack((accel_data[:, 0], accel_data[:, 1], accel_data[:, 2], gyro_data[:, 0], gyro_data[:, 1], gyro_data[:, 2], im_names[:]))
        writer.writerows(data.T)

        # for t in times:
        #     curr_position = path.get_position(t)  # x, y, z
        #     curr_orientation = euler_from_quat(path.get_orientation(t)) # qx, qy, qz, qw
        #     if p_position is None or pp_position is None:
        #         pp_position = p_position
        #         p_position = curr_position
        #         pp_orientation = p_orientation
        #         p_orientation = curr_orientation
        #         continue
        #     accel_data = ImuUtils.cal_linear_acc(
        #         [curr_position[0], p_position[0], pp_position[0]],
        #         [curr_position[1], p_position[1], pp_position[1]],
        #         [curr_position[2], p_position[2], pp_position[2]]
        #     )

        #     gyro_data = ImuUtils.cal_angular_vel(
        #         [curr_orientation[0], p_orientation[0], pp_orientation[0]],
        #         [curr_orientation[1], p_orientation[1], pp_orientation[1]],
        #         [curr_orientation[2], p_orientation[2], pp_orientation[2]]
        #     )
            
            # # Can add images to this
            # row_data = [accel_data[0], accel_data[1], accel_data[2],
            #             gyro_data[0], gyro_data[1], gyro_data[2]]
            # writer.writerow(row_data)
            
            # pp_position = p_position
            # p_position = curr_position
            # pp_orientation = p_orientation
            # p_orientation = curr_orientation


generate_training_data(Circle(center = np.array([0, 0, 0]), diameter=10.0))