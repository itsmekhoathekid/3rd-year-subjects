import torch 
from torch import nn

class ResidualConnection(nn.Module):
    def __init__(self, in_dim : int, hidden_size : int): # neu muon doi anh mau thi thay int_dim = 3
        super().__init__()

        self.conv3x3 = nn.Conv2d(
            in_channels = in_dim, # so chieu di vao/ anh 28 x 28 x 1 
            out_channels = hidden_size, # 256
            kernel_size = 3,
            padding=1
        )

        self.batchnorm_1 = nn.BatchNorm2d(
            num_features=hidden_size
        )

        self.relu_1 = nn.ReLU()

        self.conv3x3_2 = nn.Conv2d(
            in_channels = hidden_size, # so chieu di vao/ anh 28 x 28 x 1 
            out_channels = hidden_size,
            kernel_size = 3,
            padding = 1
        )

        self.batchnorm_2 = nn.BatchNorm2d(
            num_features=hidden_size
        )

        self.relu_2 = nn.ReLU()

    def forward(self, x):
        out = self.conv3x3(x)

        out = self.batchnorm_1(out)
        out = self.relu_1(out)

        out = self.conv3x3_2(out)
        out = self.batchnorm_2(out)

        out = out + x 
        out = self.relu_2(out)
        return out

class ResidualConnectionWithConv(nn.Module):
    def __init__(self,in_dim : int, hidden_size : int):
        super().__init__()

        self.conv3x3 = nn.Conv2d(
            in_channels = in_dim, # so chieu di vao/ anh 28 x 28 x 1 
            out_channels = hidden_size,
            kernel_size = 3,
            padding=1
        )

        self.batchnorm_1 = nn.BatchNorm2d(
            num_features= hidden_size
        )

        self.relu_1 = nn.ReLU()

        self.conv3x3_2 = nn.Conv2d(
            in_channels = hidden_size, # so chieu di vao/ anh 28 x 28 x 1 
            out_channels = hidden_size,
            kernel_size = 3,
            padding = 1
        )

        self.batchnorm_2 = nn.BatchNorm2d(
            num_features=hidden_size
        )

        self.conv1x1 = nn.Conv2d(
            in_channels = in_dim,
            out_channels = hidden_size,
            kernel_size = 1,
            padding = 0
        )

        self.relu_2 = nn.ReLU()

    def forward(self, x):

        out = self.conv3x3(x)

        out = self.batchnorm_1(out)
        out = self.relu_1(out)

        out = self.conv3x3_2(out)
        out = self.batchnorm_2(out)

        x = self.conv1x1(x)

        out = out + x 
        out = self.relu_2(out)

        return out

class RestNet18(nn.Module):
    def __init__(self, in_dim : int, hidden_size : int, out_dim : int):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels = in_dim,
            out_channels = hidden_size,
            kernel_size = 7,
            stride = 2,
            padding = 3
        )

        self.batchnorm_1 = nn.BatchNorm2d(
            num_features = hidden_size
        )
        
        self.relu_1 = nn.ReLU()

        self.maxpool = nn.MaxPool2d(
            kernel_size = 3,
            stride = 2,
            padding = 1
        )

        self.res_conv_1 = nn.ModuleList([
            ResidualConnection(
                in_dim=hidden_size,
                hidden_size=hidden_size
            ),
            ResidualConnection(
                in_dim=hidden_size,
                hidden_size=hidden_size
            )
        ])
        # nn.Sequential()
        self.res_conv_2 = nn.ModuleList([
            ResidualConnectionWithConv(
                in_dim=hidden_size,
                hidden_size=hidden_size
            ),
            ResidualConnection(
                in_dim=hidden_size,
                hidden_size=hidden_size
            ),
            ResidualConnectionWithConv(
                in_dim=hidden_size,
                hidden_size=hidden_size
            ),
            ResidualConnection(
                in_dim=hidden_size,
                hidden_size=hidden_size
            ),
            ResidualConnectionWithConv(
                in_dim=hidden_size,
                hidden_size=hidden_size
            ),
            ResidualConnection(
                in_dim=hidden_size,
                hidden_size=hidden_size
            )
        ])

        self.global_AvgPool = nn.AdaptiveAvgPool2d(
            output_size=(1,1)
        )

        self.fc = nn.Linear(
            in_features = hidden_size,
            out_features = out_dim
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.batchnorm_1(out)
        out = self.relu_1(out)
        out = self.maxpool(out)

        # out = self.res_conv_1(out) neu dung nn.Sequential
        for res in self.res_conv_1:
            out = res(out)

        for res in self.res_conv_2:
            out = res(out)
        
        out = self.global_AvgPool(out)
        out = out.view(out.size(0), -1) 

        out = self.fc(out)

        return out




# Corrected input tensor shape (batch_size, channels, height, width)
x = torch.rand(64, 1, 28, 28)  # Updated the shape to match expected input for Conv2d layers

from torch.nn import functional as F

class Inception1(nn.Module):
    # c1--c4 are the number of output channels for each branch
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception1, self).__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # Branch 2
        self.b2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.b2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # Branch 3
        self.b3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.b3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # Branch 4
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        b1 = F.relu(self.b1_1(x))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        b4 = F.relu(self.b4_2(self.b4_1(x)))
        return torch.cat((b1, b2, b3, b4), dim=1)
    
class GoogLeNet1(nn.Module):
    def __init__(self, in_dim, initial_hidden_size, out_dim):  # hidden size = 64
        super().__init__()

        self.block_0 = nn.Sequential(
            nn.Conv2d(in_dim, initial_hidden_size, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block_00 = nn.Sequential(
            nn.Conv2d(initial_hidden_size, initial_hidden_size, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_hidden_size, initial_hidden_size*3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        a = (initial_hidden_size * 3) // 2
        b = (initial_hidden_size * 3) // 12
        self.block_1 = nn.Sequential(
            Inception1(192, initial_hidden_size, (a, initial_hidden_size * 2), (b, initial_hidden_size // 2), initial_hidden_size // 2),
            Inception1(256, initial_hidden_size * 2, (initial_hidden_size * 2, initial_hidden_size * 3), (initial_hidden_size // 2, a), initial_hidden_size),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.block_2 = nn.Sequential(
            Inception1(480,initial_hidden_size * 3, (initial_hidden_size, initial_hidden_size * 3 + initial_hidden_size // 4), (initial_hidden_size // 4, initial_hidden_size // 2 + initial_hidden_size // 4), initial_hidden_size),
            Inception1(512, initial_hidden_size * 2 + initial_hidden_size // 2, (initial_hidden_size * 2 - initial_hidden_size // 4, initial_hidden_size * 4 - initial_hidden_size // 2), (initial_hidden_size // 4 + initial_hidden_size // 8, initial_hidden_size), initial_hidden_size),
            Inception1(512, initial_hidden_size * 2, (initial_hidden_size * 2, initial_hidden_size * 4), (initial_hidden_size // 4 + initial_hidden_size // 8, initial_hidden_size), initial_hidden_size),
            Inception1(512,initial_hidden_size * 2 - initial_hidden_size // 4, (initial_hidden_size * 2 + initial_hidden_size // 4, initial_hidden_size * 4 + initial_hidden_size // 2), (initial_hidden_size // 2, initial_hidden_size), initial_hidden_size),
            Inception1(528,initial_hidden_size * 4, (initial_hidden_size * 2 + initial_hidden_size // 2, initial_hidden_size * 5), (initial_hidden_size // 2, initial_hidden_size * 2), initial_hidden_size * 2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block_3 = nn.Sequential(
            Inception1(832, initial_hidden_size * 4, (initial_hidden_size * 2 + initial_hidden_size // 2, initial_hidden_size * 5), (initial_hidden_size // 2, initial_hidden_size * 2), initial_hidden_size * 2),
            Inception1(832, initial_hidden_size * 6, (initial_hidden_size * 3, initial_hidden_size * 6), (initial_hidden_size // 2 + initial_hidden_size // 4, initial_hidden_size * 2), initial_hidden_size * 2),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )

        self.fc = nn.Linear(in_features=initial_hidden_size * 16, out_features=out_dim)

    def forward(self, x):
        out = self.block_0(x)
        out = self.block_00(out)
        #print(f'After block_0 {out.shape}')
        out = self.block_1(out)
        #print(f'After block_1: {out.shape}')
        out = self.block_2(out)
        #print(f'After block_2: {out.shape}')
        out = self.block_3(out)
        #print(f'After block_3: {out.shape}')
        out = self.fc(out)
        #print(f'Final output shape: {out.shape}')
        # raise ValueError('Stop here')
        return out



class LeNet(nn.Module):
    def __init__(self, int_dim, out_dim):
        super(LeNet, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=int_dim, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate the input size for the first fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),  # Adjusted input size
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, out_dim)
        )

    def forward(self, x):
        out = self.block(x)
        out = out.view(out.size(0), -1)  # Flatten the output
        out = self.fc(out)
        return out
