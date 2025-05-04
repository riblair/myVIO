import torch
import torch.nn as nn
import numpy as np

import Models.utils as utils


class InertialOdometry(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, output_size=7):
        super(InertialOdometry, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        # self.fc1 = nn.Linear(hidden_size, 128)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.prelu = nn.PReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc1(x[:, -1, :])
        x = self.prelu(x)
        x = self.fc2(x)
        return x
    
    def loss(self, gt: torch.Tensor, measured: torch.Tensor):
        pos_hat, orient_hat = measured[:, :3], measured[:, 3:]
        pos, orient = gt[:, :3], gt[:, 3:]
        
        orient_hat = torch.nn.functional.normalize(orient_hat, p=2, dim=1)
        orient = torch.nn.functional.normalize(orient, p=2, dim=1)
        
        loss_fn = nn.L1Loss()
        pos_loss = loss_fn(pos_hat, pos)
        orient_loss = utils.geodesic_loss(utils.quaternion_to_matrix(orient_hat), utils.quaternion_to_matrix(orient), reduction='mean')
        
        return pos_loss + orient_loss
