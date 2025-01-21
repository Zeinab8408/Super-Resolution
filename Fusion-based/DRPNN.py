
### DRPNN : Boosting the Accuracy of Multispectral Image Pansharpening by Learning a Deep Residual Network ###
### doi : 10.1109/LGRS.2017.2736020 ###
### Implementation Link: https://github.com/matciotola ###

class DRPNN(nn.Module):
    def __init__(self):
        in_channels = hs_channels + PAN_channels
        super(DRPNN, self).__init__()
        self.Conv_1 = nn.Conv2d(in_channels, 64, 7, padding=(3, 3))
        self.Conv_2 = nn.Conv2d(64, 64, 7, padding=(3, 3))
        self.Conv_3 = nn.Conv2d(64, 64, 7, padding=(3, 3))
        self.Conv_4 = nn.Conv2d(64, 64, 7, padding=(3, 3))
        self.Conv_5 = nn.Conv2d(64, 64, 7, padding=(3, 3))
        self.Conv_6 = nn.Conv2d(64, 64, 7, padding=(3, 3))
        self.Conv_7 = nn.Conv2d(64, 64, 7, padding=(3, 3))
        self.Conv_8 = nn.Conv2d(64, 64, 7, padding=(3, 3))
        self.Conv_9 = nn.Conv2d(64, 64, 7, padding=(3, 3))
        self.Conv_10 = nn.Conv2d(64, in_channels, 7, padding=(3, 3))
        self.Conv_11 = nn.Conv2d(in_channels, in_channels - PAN_channels, 3, padding=(1, 1))
    def forward(self, LRHS, PAN):
        w = PAN.shape[2]
        HRHS = F.interpolate(LRHS, size=[w,w], mode='bicubic', align_corners=True)
        x = torch.cat((HRHS, PAN), dim=1)
        x1 = F.relu(self.Conv_1(x))
        x2 = F.relu(self.Conv_2(x1))
        x3 = F.relu(self.Conv_3(x2))
        x4 = F.relu(self.Conv_4(x3))
        x5 = F.relu(self.Conv_5(x4))
        x6 = F.relu(self.Conv_6(x5))
        x7 = F.relu(self.Conv_7(x6))
        x8 = F.relu(self.Conv_8(x7))
        x9 = F.relu(self.Conv_9(x8))
        x10 = self.Conv_10(x9)
        x11 = self.Conv_11(F.relu(x10 + x))
        return x11

net = DRPNN().to(device)
optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4) #, weight_decay=0
loss_fn = nn.MSELoss().to(device)
