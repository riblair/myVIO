import torch.nn as nn
import torch
import numpy as np
from utils import *

class VisualInertialOdometry(nn.Module):
    def __init__(self, input_dim_inertial=6, hidden_dim_inertial=256, num_layers=2, memory_size=100, dropout=0.5, learning_rate=1e-3):
        super().__init__()
        
        """ Visual Odom Part """
        self.fnet = nn.Sequential(
            BasicConvEncoder(output_dim=256, norm_fn='instance', dropout=dropout),
            POLAUpdate(embed_dim=256, depth=6, num_head=8, window_size=7, neig_win_num=1)
        )
        self.hidden_dim_inertial = hidden_dim_inertial 
        self.conv_reduce = nn.Conv2d(256, 128, kernel_size=1)
        
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d((22, 28))
        self.relu = nn.ReLU()
        
        
        """ Multihead attention after concat """
        self.cross_attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=dropout)
        
        self.conv_attn_map = nn.Conv2d(256, 32, kernel_size=1)

        """ Inertial Odometry Part """
        self.lstm_inertial = nn.LSTM(input_size=input_dim_inertial,
                            hidden_size=hidden_dim_inertial,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=True,
                            batch_first=True)

        
        self.fc1_position = nn.Linear(19840, 256)
        self.fc2_position = nn.Linear(256, 3)
        self.fc1_orientation = nn.Linear(19840, 256)
        self.fc2_orientation = nn.Linear(256, 4)

    def forward(self, image1, image2, imu):
        """ Visual Odom """
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        
        fmap1, fmap2 = self.fnet([image1, image2])
        
        # Reduce dimensionality
        fmap1 = self.conv_reduce(self.adaptive_max_pool(fmap1))
        fmap2 = self.conv_reduce(self.adaptive_max_pool(fmap2))

        # Compute sequence length from spatial dimensions for concatenation
        sequence_length = fmap1.shape[2] * fmap1.shape[3]  # width * height
        fmap1 = fmap1.view(fmap1.shape[0], fmap1.shape[1], sequence_length)
        fmap2 = fmap2.view(fmap2.shape[0], fmap2.shape[1], sequence_length)

        # Concatenate feature maps along the channel dimension
        combined_fmaps = torch.cat([fmap1, fmap2], dim=1)  # Concatenate on channel dimension
        combined_fmaps = combined_fmaps.permute(2, 0, 1)  # Reshape for sequence model: (sequence_length, batch_size, embedding_dim)
        
        """ Inertial Odom """
        # Process inertial information with LSTM
        _, (hidden_inertial, _) = self.lstm_inertial(imu)
        
        # Concat features to combine for VIO
        concatenated_features = torch.cat([combined_fmaps, hidden_inertial], dim=0)
        # Cross-attention
        attn_output, _ = self.cross_attention(concatenated_features, concatenated_features, concatenated_features)
        
        attn_output = attn_output.permute(1, 2, 0).reshape(-1, 256, 20, 31)
        attn_output = self.conv_attn_map(attn_output)

        # Flatten the output from attention
        attn_output = attn_output.contiguous().view(attn_output.size(0), -1)
        
        # Position pathway
        pos = self.relu(self.fc1_position(attn_output))
        pos = self.fc2_position(pos)

        # Orientation pathway
        orient = self.relu(self.fc1_orientation(attn_output))
        orient = self.fc2_orientation(orient)
        
        # Combine position and orientation into a single output tensor
        res = torch.cat([pos, orient], dim=1)
    
        return res
    
    def loss(self, gt: torch.Tensor, measured: torch.Tensor):
        pos_hat, orient_hat = measured[:, :3], measured[:, 3:]
        pos, orient = gt[:, :3], gt[:, 3:]
        
        orient_hat = torch.nn.functional.normalize(orient_hat, p=2, dim=1)
        orient = torch.nn.functional.normalize(orient, p=2, dim=1)
        
        pos_loss = nn.L1Loss(pos_hat, pos)
        orient_loss = geodesic_loss(quaternion_to_matrix(orient_hat), quaternion_to_matrix(orient))
        
        return pos_loss + orient_loss
