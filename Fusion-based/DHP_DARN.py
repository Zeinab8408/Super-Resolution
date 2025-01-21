### DHP-DARN  : Hyperspectral Pansharpening Using Deep Prior and Dual Attention Residual Network ###
### doi: 10.1109/TGRS.2020.2986313 ###
### Implementation Link : https://github.com/yxzheng24 ###

class CSA(nn.Module):
    def __init__(self, in_channels):
        super(CSA, self).__init__()
        self.in_channels = in_channels
        r = 16 # Downsampling ratio of the CA modele
        # Input feature extraction
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, padding=1)
        # CA
        self.gap   = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv3 = nn.Conv2d(in_channels=self.in_channels, out_channels=int(self.in_channels/r), kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=int(self.in_channels/r), out_channels=self.in_channels, kernel_size=1)
        # SA
        self.conv5 = nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=1)
        # Sigmoid
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        u       = self.conv2(F.relu(self.conv1(x)))
        M_CA    = self.sigmoid(self.conv4(F.relu(self.conv3(self.gap(u)))))
        M_SA    = self.sigmoid(self.conv5(u))
        U_CA    = u*M_CA
        U_SA    = u*M_SA
        out     = U_CA + U_SA + x
        return out

class DHP_DARN(nn.Module):
    def __init__(self):
        super(DHP_DARN, self).__init__()
        self.is_DHP_MS      = False
        self.in_channels    = ms_channels
        self.out_channels   = ms_channels
        self.N_Filters      = 64
        # self.N_modules      = config["N_modules"]
        # FEN Layer
        self.FEN    = nn.Conv2d(in_channels=self.in_channels+ pan_channels, out_channels=self.N_Filters, kernel_size=3, padding=1)
        # CSA RESBLOCKS
        self.CSA1   = CSA(in_channels=self.N_Filters)
        self.CSA2   = CSA(in_channels=self.N_Filters)
        self.CSA3   = CSA(in_channels=self.N_Filters)
        self.CSA4   = CSA(in_channels=self.N_Filters)
        #RNN layer
        self.RRN    = nn.Conv2d(in_channels=self.N_Filters, out_channels=self.out_channels, kernel_size=3, padding=1)
    def forward(self, X_MS, X_PAN):
        if not self.is_DHP_MS:
            X_MS_UP = F.interpolate(X_MS, scale_factor=(4,4), mode ='bilinear')
        else:
            X_MS_UP = X_MS
        # Concatenating the generated H_UP with P
        x = torch.cat((X_MS_UP, X_PAN), dim=1)
        # FEN
        x = self.FEN(x)
        # DARN
        x = self.CSA1(x)
        x = self.CSA2(x)
        x = self.CSA3(x)
        x = self.CSA4(x)
        # RRN
        x = self.RRN(x)
        # Final output
        x = x + X_MS_UP
        # output = {"pred": x}
        return x

net = DHP_DARN().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3) #, weight_decay=0

def loss_fn(x,y):
    mae = nn.L1Loss().to(device)
    loss = mae(y, x)
    return loss
  
