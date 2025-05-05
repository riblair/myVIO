import argparse
import random
import csv
import os

import torch
from torch.optim import Adam
import numpy as np
import cv2

from Models.InertialOdometry import InertialOdometry as IO
from Models.VisualOdometry import VisualOdometry as VO
from Models.VisualInertialOdometry import VisualInertialOdometry as VIO

def main():
    checkpoint_filepath = 'Code/Phase2/checkpoints/180a99model.ckpt' #path to checkpoint
    model = IO()
    checkpoint = torch.load(checkpoint_filepath)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Import IMU state to test
    imu_state_filepath = 'Code/Phase2/Data/Test/traj_0.csv'
    # send states though model in groups of 10
    with open(imu_state_filepath, "r") as imu_file:
        reader = csv.reader(imu_file)
        rows = [r for r in reader]
    
    imu_batches = []
    imu_measurements = []
    for row_idx in range(len(rows)):
        data_row = rows[row_idx]
        if row_idx % 10 == 0 and row_idx != 0:
            imu_batches.append(imu_measurements)
            imu_measurements = []
        imu_measurements.append(data_row[0:6])
    
    delta_poses = []
    for imu_block in imu_batches:
        imu_np = np.array([imu_block], dtype=np.float32)
        imu_data = torch.from_numpy(imu_np)
        estimated_pose = model(imu_data)
        estimated_pose_np = estimated_pose.detach().numpy().flatten()
        delta_poses.append(estimated_pose_np.tolist())
    delta_poses = np.array(delta_poses)
        
    # Use dead-reckoning to determine full trajectory
    with open('final_trajectory.csv', "w") as file:
        writer = csv.writer(file)
        world_pose = np.zeros((1,7))
        for row_idx in range(delta_poses.shape[0]):
            delta_pose = delta_poses[row_idx, :]
            world_pose += delta_pose
            # record estimated pose in csv file
            list_to_write = [
                world_pose[0,0], world_pose[0,1], world_pose[0,2], world_pose[0,3],
                world_pose[0,4], world_pose[0,5], world_pose[0,6]
            ]
            writer.writerow(list_to_write)
    return

if __name__ == '__main__':
    main()
