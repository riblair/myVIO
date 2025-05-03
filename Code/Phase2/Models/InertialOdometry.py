import torch
import torch.nn as nn
import numpy as np


class InertialOdometry(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, output_size=6):
        super(InertialOdometry, self).__init__()
        #############################
        # network initialization
        #############################
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        # self.fc1 = nn.Linear(hidden_size, 128)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.prelu = nn.PReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        #############################
        # network structure
        #############################
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.fc1(x[:, -1, :])
        x = self.prelu(x)
        x = self.fc2(x)
        return x
