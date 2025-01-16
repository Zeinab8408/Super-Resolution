from math import ceil
    
### HyperPNN3 IMPLEMENTATION ###
class HyperPNN(nn.Module):
    def __init__(self,hs_channels):
        super(HyperPNN, self).__init__()
        self.is_DHP_MS      = False
        self.in_channels    = hs_channels
        self.out_channels   = hs_channels
        self.factor         = 4
        self.mid_channels = 64
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.mid_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=self.mid_channels+pan_channels, out_channels=self.mid_channels, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=3, padding=1, stride=1)
        self.conv6 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=1)
        self.conv7 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.out_channels, kernel_size=1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, X_MS,X_PAN):
        if not self.is_DHP_MS:
            X_MS_UP = F.interpolate(X_MS, scale_factor=(self.factor,self.factor),mode ='bilinear')
        else:
            X_MS_UP = X_MS
        x = F.relu(self.conv1(X_MS_UP))
        x = F.relu(self.conv2(x))
        x = torch.cat((x, X_PAN), dim=1)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.conv7(x)
        x = x+X_MS_UP

        return x

net = HyperPNN(hs_channels).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001) #, weight_decay=0

def loss_fn(x,y):
    loss = torch.norm((y-x), p="fro")
    return loss
