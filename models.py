import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

### HyperPNN: Hyperspectral Pansharpening via Spectrally Predictive Convolutional Neural Networks ###
### doi: 10.1109/JSTARS.2019.2917584 ###
class HyperPnn(nn.Module):
    def __init__(self,hs_channels):
        super(HyperPNN, self).__init__()
        self.HSchannels   = hs_channels
        self.PANchannels  = 1
        self.mid_channels = 64
        self.conv1 = nn.Conv2d(in_channels=self.HSchannels, out_channels=self.mid_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=self.mid_channels + self.PANchannels, out_channels=self.mid_channels, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=3, padding=1, stride=1)
        self.conv6 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=1)
        self.conv7 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.HSchannels, kernel_size=1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, LRHS,PAN):
        HRHS = F.interpolate(LRHS, size=[PAN.shape[1],PAN.shape[1]],mode ='bilinear')
        x = F.relu(self.conv1(HRHS))
        x = F.relu(self.conv2(x))
        x = torch.cat((x,PAN), dim=1)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.conv7(x)
        x =  x + HRHS
        return x

net = HyperPnn(hs_channels).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001) # weight_decay=0

def loss_fn(x,y):
    loss = torch.norm((y-x), p="fro")
    return loss
