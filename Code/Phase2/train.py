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

def generate_IO_batch(batch_size: int, train=True):
    # Need to return (batchsize, 10, 6) set of imu data points
    batch = []
    ground_truth = []
    csv_rows = []
    num_trajectories = 0
    directory_filepath = 'Code/Phase2/data/train' if train else 'Code/Phase2/data/val'
    for _, _, files in os.walk(directory_filepath):
        num_trajectories += len(files)
    traj_idx = random.randint(1, num_trajectories-1)
    # traj_idx = 1
    data_filepath = directory_filepath + f'/traj_{traj_idx}.csv'
    gt_filepath = f'Code/Phase2/groundtruth/train/traj_{traj_idx}.csv' if train else f'Code/Phase2/groundtruth/val/traj_{traj_idx}.csv'
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

def generate_VO_batch(batch_size: int):
    batch = []
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
            batch.append([image_1, image_2])
    batch_np = np.array(batch)
    return batch_np

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
    elif args.mode == 'VO':
        model = VO()
        weights = torch.load("Code/Phase2/weights/gmflownet-kitti.pth", map_location='cpu')
        for key in weights.keys():
            if key.replace('module.', '') in model.state_dict().keys():
                model.state_dict()[key.replace('module.', '')] = weights[key]
        optimizer = Adam(lr=0.001)
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
    save_rate = 20
    
    for epoch in range(start_epoch, epochs):
        num_iterations_per_epoch = 100
        epoch_loss = 0
        for per_epoch_counter in range(num_iterations_per_epoch):
            if args.mode == 'IO':
                batch, ground_truth = generate_IO_batch(batch_size)
            elif args.mode == 'VO':
                batch = generate_VO_batch(batch_size)
            elif args.mode == 'VIO':
                batch = generate_VIO_batch(batch_size)
            
            estimated_pose = model(batch)
            training_loss = model.loss(ground_truth, estimated_pose) # TODO: Read ground truth and import estimated pose
            print(training_loss)
            epoch_loss += training_loss
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()
        print(f"EPOCH LOSS: {epoch_loss}")
        print()
            
            # Save checkpoint every some SaveCheckPoint's iterations
            # if per_epoch_counter % save_checkpoint == 0:
            #     # Save the Model learnt in this epoch
            #     SaveName = (
            #         "Code/Phase2/checkpoints/"
            #         + str(epoch)
            #         + "a"
            #         + str(per_epoch_counter)
            #         + "model.ckpt"
            #     )

            #     torch.save(
            #         {
            #             "epoch": epoch,
            #             "model_state_dict": model.state_dict(),
            #             "optimizer_state_dict": optimizer.state_dict(),
            #             "loss": trainging_loss,
            #         },
            #         SaveName,
            #     )
            #     print("\n" + SaveName + " Model Saved...")
                
        # with torch.no_grad():
        #     if args.mode == 'IO':
        #         batch, ground_truth = generate_IO_batch(batch_size, train=False)
        #     elif args.mode == 'VO':
        #         batch = generate_VO_batch(batch_size)
        #     elif args.mode == 'VIO':
        #         batch = generate_VIO_batch(batch_size)
        #     val_pose = model(batch)
        #     loss = model.loss(ground_truth, val_pose)
        # print(f"Validation Loss: {loss}, Training Loss: {trainging_loss}")
        if epoch % save_rate == 0:
            SaveName = (
                        "Code/Phase2/checkpoints/"
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
    args = parser.parse_args()
    main(args)
