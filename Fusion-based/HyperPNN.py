### HyperPNN: Hyperspectral Pansharpening via Spectrally Predictive Convolutional Neural Networks ###
### doi: 10.1109/JSTARS.2019.2917584 ###

class HyperPnn(nn.Module):
    def __init__(self,hs_channels, PAN_channels):
        super(HyperPnn, self).__init__()
        mid_channels = 64
        self.conv1 = nn.Conv2d(hs_channels, mid_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(mid_channels + PAN_channels, mid_channels, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, stride=1)
        self.conv6 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)
        self.conv7 = nn.Conv2d(mid_channels, hs_channels, kernel_size=1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, LRHS,PAN): # LRHS-Size = [1,HS_channels,width,height],  PAN-size = [1,1,width,height]
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

net = HyperPnn(hs_channels, PAN_channels).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001) # weight_decay=0

def lossFn(x,y):
    loss = torch.norm((y-x), p="fro")
    return loss
