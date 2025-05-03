import argparse
import torch
from torch.optim import Adam

from Models.InertialOdometry import InertialOdometry as IO
from Models.VisualOdometry import VisualOdometry as VO
from Models.VisualInertialOdometry import VisualInertialOdometry as VIO

def main(args):
    
    # Waiting until we have dataset functionality before implementing full training script.
    
    if args.mode == 'IO':
        model = IO()
        optimizer = Adam()
        lr = 0.001
        epochs = 200
        batch_size = 32
    elif args.mode == 'VO':
        model = VO()
        weights = torch.load("Code/Phase2/weights/gmflownet-kitti.pth", map_location='cpu')
        for key in weights.keys():
            if key.replace('module.', '') in model.state_dict().keys():
                model.state_dict()[key.replace('module.', '')] = weights[key]
        optimizer = Adam()
        lr = 0.0001
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
        optimizer = Adam()
        lr = 0.001
        epochs = 50
        batch_size = 16
    else:
        raise ValueError(f"Unknown odometry mode. Expected 'IO', 'VO', or 'VIO'. Instead got {args.mode}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train models for different modalities")
    parser.add_argument('--data_file', type=str, required=True, help='Path to the data file')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training')
    parser.add_argument('--mode', type=str, choices=['VO', 'IO', 'VIO'], required=True, help='Mode of operation: VO, IO, or VIO')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 training')
    args = parser.parse_args()
    main(args)
