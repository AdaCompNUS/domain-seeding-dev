import torch
import torchvision
from torch import nn
from models.base import BaseNetwork


class ISModel(BaseNetwork):
    '''
    A simple system identification model that encodes a stack of image observations,
    concatenate the feature with the hypothetical system parameter set, xi_h,
    and predict (xi_gt - xi_h), where xi_gt is the ground truth parameter.
    '''
    def __init__(self, time_channels=8, num_param=5):
        super(ISModel, self).__init__()
        self.flatten = nn.Flatten()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=time_channels, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3,3), stride=(2,2), padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(output_size=(1, 1))
        )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1024 + num_param, 128),
            nn.ReLU(),
            nn.Linear(128, num_param),
        )
        print(self)

    def forward(self, x, hypo_xi):
        '''
        x: A sequence of observations, shape (b,T,h,w)
        hypo_xi: the hypothetical system parameter for the sequence of observations
        '''
        features = self.cnn(x).flatten(1)
        features = torch.cat((features, hypo_xi), dim=1)
        logits = self.linear_relu_stack(features)
        return logits


if __name__ == '__main__':
    model = ISModel()
    print(model)
    print(model(torch.randn(2, 8, 112, 112), torch.rand(2, 5)).shape)