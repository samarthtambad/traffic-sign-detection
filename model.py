import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

# https://pytorch.org/docs/stable/nn.html?highlight=conv2d#torch.nn.Conv2d
# https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3)    # add dropout
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3)   # add dropout
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)  # add dropout
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3) # add dropout
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3) # add dropout
        self.conv8 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv9 = nn.Conv2d(128, nclasses, kernel_size=1)
        self.conv1_drop = nn.Dropout2d()
        self.conv2_drop = nn.Dropout2d()
        self.conv4_drop = nn.Dropout2d()
        self.conv5_drop = nn.Dropout2d()
        self.conv7_drop = nn.Dropout2d()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        x = self.stn(x)
        x = F.leaky_relu(self.conv1_drop(self.conv1(x)))
        x = F.leaky_relu(self.conv2_drop(self.conv2(x)))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4_drop(self.conv4(x)))
        x = F.leaky_relu(self.conv5_drop(self.conv5(x)))
        x = F.leaky_relu(self.conv6(x))
        x = F.leaky_relu(self.conv7_drop(self.conv7(x)))
        x = F.leaky_relu(self.conv8(x))
        x = F.leaky_relu(self.conv9(x))
        # x = x.view(-1, 512 * 1 * 1)
        # global averaging over 6 × 6 spatial dimensions
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(-1, nclasses)
        return F.log_softmax(x, dim=1)

