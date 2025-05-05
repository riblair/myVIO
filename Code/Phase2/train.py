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

def generate_IO_batch(batch_size: int, path_type:str, train=True):
    # Need to return (batchsize, 10, 6) set of imu data points
    batch = []
    ground_truth = []
    csv_rows = []
    num_trajectories = 0
    directory_filepath = f'Code/Phase2/Data/{path_type}/Train' if train else f'Code/Phase2/Data/{path_type}/Val'
    for _, _, files in os.walk(directory_filepath):
        num_trajectories += len(files)
    traj_idx = random.randint(1, num_trajectories-1)
    # traj_idx = 1
    data_filepath = directory_filepath + f'/path.csv'
    gt_filepath = directory_filepath + '/gt_data.csv'
    with open(data_filepath, "r") as file:
        reader = csv.reader(file)
        rows =[r for r in reader]
    with open(gt_filepath, "r") as file:
        reader = csv.reader(file)
        gt_rows = [r for r in reader]
    num_data_rows = len(rows)
    # subtract 10 since we always want 10 datapoints. prevents index out of bounds error
    for _ in range(batch_size):
        starting_idx = random.randint(0, num_data_rows-10)
        batch_element = []
        csv_rows.append((starting_idx, starting_idx+10))
        for i in range(starting_idx, starting_idx+10):
            data_row = rows[i]
            batch_element.append(data_row[0:6])  # input row as a row, so batch is a 2d array
        batch.append(batch_element)
        ground_truth.append(gt_rows[starting_idx])
    batch_np = np.array(batch, dtype=np.float32)
    ground_truth = np.array(ground_truth, dtype=np.float32)
    return torch.from_numpy(batch_np), torch.from_numpy(ground_truth)

def generate_VO_batch(batch_size: int, train=True):
    batch = []
    ground_truth = []
    num_trajectories = 0
    directory_filepath = 'Code/Phase2/Data/Train/Images' if train else 'Code/Phase2/Data/Val/Images'
    for _, _, files in os.walk(directory_filepath):
        num_trajectories = len(files)
    traj_idx = random.randint(1, num_trajectories)
    for _, _, files in os.walk(directory_filepath):
        num_trajectories += len(files)
    traj_idx = random.randint(1, num_trajectories-1)
    
    gt_filepath = f'Code/Phase2/groundtruth/train/gt_data.csv' if train else f'Code/Phase2/groundtruth/val/gt_data.csv'
    with open(gt_filepath, "r") as file:
        reader = csv.reader(file)
        gt_rows = [r for r in reader]
    num_images = len([entry for entry in os.scandir(directory_filepath) if entry.is_file()])
    batch_np = np.array(batch)
    for _ in range(batch_size):
        starting_idx = random.randint(0, num_images-10)
        batch_element = []
        data_filepath = directory_filepath + f'/im_{starting_idx:05d}.png'
        image_1 = cv2.imread(data_filepath)
        data_filepath = directory_filepath + f'/im_{starting_idx+10:05d}.png'
        image_2 = cv2.imread(data_filepath)
        image_stack = np.concat((image_1, image_2), axis=2)  # Stack along channel dimension
        batch.append(image_stack)
        ground_truth.append(gt_rows[starting_idx])
    batch_np = np.array(batch, dtype=np.float32)
    ground_truth = np.array(ground_truth, dtype=np.float32)
    return torch.from_numpy(batch_np), torch.from_numpy(ground_truth)
    # return batch_np

def generate_VIO_batch(batch_size):
    image_batch = []
    imu_batch = []
    num_trajectories = 0
    for _, _, files in os.walk('data'):
        num_trajectories += len(files)
    traj_idx = random.randint(1, num_trajectories)
    data_filepath = f'data/traj_{traj_idx}.csv'
    with open(data_filepath, "r") as file:
        reader = csv.reader(file)
        num_rows = len(reader)
        for _ in range(batch_size):
            starting_idx = random.randint(0, num_rows-10)
            # TODO: Update with image directory filepath
            image_1 = cv2.imread(reader[starting_idx][-1])
            image_2 = cv2.imread(reader[starting_idx+10][-1])
            image_batch.append([image_1, image_2])
            imu_batch_element = []
            for i in range(starting_idx, starting_idx+10):
                row = reader[i]
                imu_batch_element.append(row)  # input row as a row, so batch is a 2d array
            imu_batch.append(imu_batch_element)
    imu_batch_np = np.array(imu_batch)
    return torch.from_numpy(imu_batch_np), image_batch

def main(args):
    
    # Waiting until we have dataset functionality before implementing full training script.
    
    if args.mode == 'IO':
        model = IO()
        optimizer = Adam(model.parameters(), lr=0.001)
        epochs = 200
        batch_size = 32
        save_rate = 20
    elif args.mode == 'VO':
        model = VO()
        optimizer = Adam(model.parameters(), lr=0.001)
        weights = torch.load("Code/Phase2/weights/gmflownet-kitti.pth", map_location='cpu')
        for key in weights.keys():
            if key.replace('module.', '') in model.state_dict().keys():
                model.state_dict()[key.replace('module.', '')] = weights[key]
        epochs = 50
        batch_size = 16
    elif args.mode == 'VIO':
        model = VIO()
        weights = torch.load("Code/Phase2/weights/gmflownet-kitti.pth", map_location='cpu')
        # MUST RUN WITH TRAINED IO WEIGHTS BEFORE RUNNING THIS
        lstm_weights = torch.load("...", map_location='cpu')['state_dict']  #TODO: Update with trained weights
        for key in weights.keys():
            if key.replace('module.', '') in model.state_dict().keys():
                model.state_dict()[key.replace('module.', '')] = weights[key]
           
        for key in lstm_weights.keys():
            if key.replace('lstm.', 'lstm_inertial.') in model.state_dict().keys():
                model.state_dict()[key.replace('lstm.', 'lstm_inertial.')] = lstm_weights[key]  
        optimizer = Adam(lr=0.001)
        epochs = 50
        batch_size = 16
    else:
        raise ValueError(f"Unknown odometry mode. Expected 'IO', 'VO', or 'VIO'. Instead got {args.mode}")
    
    start_epoch = 0
    
    
    for epoch in range(start_epoch, epochs):
        num_iterations_per_epoch = 100
        epoch_loss = 0
        for per_epoch_counter in range(num_iterations_per_epoch):
            if args.mode == 'IO':
                batch, ground_truth = generate_IO_batch(batch_size, path_type=args.path_type)
                estimated_pose = model(batch)
            elif args.mode == 'VO':
                batch, ground_truth = generate_VO_batch(batch_size)
                estimated_pose = model(batch[:])
            elif args.mode == 'VIO':
                batch, ground_truth = generate_VIO_batch(batch_size)
            
            
            training_loss = model.loss(ground_truth, estimated_pose) # TODO: Read ground truth and import estimated pose
            print(training_loss)
            epoch_loss += training_loss
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()
                
        print(f"EPOCH LOSS: {epoch_loss}")
        print()
        if epoch % save_rate == 0:
            SaveName = (
                        "Code/Phase2/Checkpoints/"
                        + str(epoch)
                        + "a"
                        + str(per_epoch_counter)
                        + "model.ckpt"
                    )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": training_loss,
                },
                SaveName,
            )
            print("\n" + SaveName + " Model Saved...")
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train models for different modalities")
    parser.add_argument('--data_file', type=str, default='Code/Phase2/data/', help='Path to the data file')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training')
    parser.add_argument('--mode', type=str, default='IO', choices=['VO', 'IO', 'VIO'], help='Mode of operation: VO, IO, or VIO')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 training')
    parser.add_argument('--path_type', type=str, default='Straight_Line', choices=['Straight_Line', 'Circle', 'Sinusoid'])
    args = parser.parse_args()
    main(args)
