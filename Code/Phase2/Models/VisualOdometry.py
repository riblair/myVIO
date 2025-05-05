import torch
import torch.nn as nn
import numpy as np
import Models.utils as utils

class VisualOdometry(nn.Module):
    def __init__(self, in_channels=6, hidden_size=128, output_size=6, concat=False):
        super().__init__(VisualOdometry, self)
        self.fnet = nn.Sequential(
            utils.BasicConvEncoder(output_dim=256, norm_fn='instance', dropout=dropout),
            utils.POLAUpdate(embed_dim=256, depth=6, num_head=8, window_size=7, neig_win_num=1)
        )
        self.conv1_position = nn.Conv2d(512, 16, kernel_size=1)
        self.conv1_orientation = nn.Conv2d(512, 16, kernel_size=1)
        
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d((30, 40))
        self.relu = nn.ReLU()
        
        self.fc1_position = nn.Linear(19200, 256)
        self.fc2_position = nn.Linear(256, 3)
        self.fc1_orientation = nn.Linear(19200, 256)
        self.fc2_orientation = nn.Linear(256, 4)
    
    
    def forward(self, image1, image2):
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        fmap1, fmap2 = self.fnet([image1, image2])
        feats = torch.cat([fmap1, fmap2], dim=1)
        
        # Position pathway
        pos = self.conv1_position(feats)
        pos = self.adaptive_max_pool(pos)
        
        pos = pos.view(pos.size(0), -1)  # Flatten
        pos = self.relu(self.fc1_position(pos))
        pos = self.fc2_position(pos)

        # Orientation pathway
        orient = self.conv1_orientation(feats)
        orient = self.adaptive_max_pool(orient)
        
        orient = orient.view(orient.size(0), -1)  # Flatten
        orient = self.relu(self.fc1_orientation(orient))
        orient = self.fc2_orientation(orient)
        
        # Combine position and orientation into a single output tensor
        res = torch.cat([pos, orient], dim=1)
        
        return res
    
    def freeze_fnet(self):
        # Freeze the fnet layer to prevent updates during training. We want to keep the pre-trained weights
        for param in self.fnet.parameters():
            param.requires_grad = False
            
    def loss(self, gt: torch.Tensor, measured: torch.Tensor):
        pos_hat, orient_hat = measured[:, :3], measured[:, 3:]
        pos, orient = gt[:, :3], gt[:, 3:]
        
        orient_hat = torch.nn.functional.normalize(orient_hat, p=2, dim=1)
        orient = torch.nn.functional.normalize(orient, p=2, dim=1)
        
        loss_fn = nn.L1Loss()
        pos_loss = loss_fn(pos_hat, pos)
        orient_loss = utils.geodesic_loss(utils.quaternion_to_matrix(orient_hat), utils.quaternion_to_matrix(orient), reduction='mean')
        
        return pos_loss + orient_loss
    
