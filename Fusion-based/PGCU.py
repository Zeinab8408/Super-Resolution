### PGCU : Probability-based Global Cross-modal Upsampling for Pansharpening ###
### doi: 10.48550/arXiv.2303.13659 ###
### Implementation Link : https://github.com/Zeyu-Zhu/PGCU ###

class DownSamplingBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownSamplingBlock, self).__init__()
        self.Conv = nn.Conv2d(in_channel, out_channel, (3,3), 2, 1)
        self.MaxPooling = nn.MaxPool2d((2, 2))
    def forward(self, x):
        out = self.MaxPooling(self.Conv(x))
        return out
    
class PGCU(nn.Module):
    def __init__(self, Channel=ms_channels, VecLen=128, NumberBlocks=3):
        super(PGCU, self).__init__()
        self.BandVecLen = VecLen//Channel
        self.Channel = ms_channels
        self.VecLen = VecLen
        ## Information Extraction
        # F.size == (Vec, W, H)
        self.FPConv = nn.Conv2d(1, Channel, (3,3), 1, 1)
        self.FMConv = nn.Conv2d(Channel, Channel, (3,3), 1, 1)
        self.FConv = nn.Conv2d(Channel*2, VecLen, (3,3), 1, 1)
        # G.size == (Vec, W/pow(2, N), H/pow(2, N))
        self.GPConv = nn.Sequential()
        self.GMConv = nn.Sequential()
        self.GConv = nn.Conv2d(Channel*2, VecLen, (3,3), 1, 1)
        for i in range(NumberBlocks):
            if i == 0:
                self.GPConv.add_module('DSBlock'+str(i), DownSamplingBlock(1, Channel))
            else:
                self.GPConv.add_module('DSBlock'+str(i), DownSamplingBlock(Channel, Channel))
                self.GMConv.add_module('DSBlock'+str(i-1), DownSamplingBlock(Channel, Channel))
        # V.size == (C, W/pow(2, N), H/pow(2, N)), k=W*H/64
        self.VPConv = nn.Sequential()
        self.VMConv = nn.Sequential()
        self.VConv = nn.Conv2d(Channel*2, Channel, (3,3), 1, 1)
        for i in range(NumberBlocks):
            if i == 0:
                self.VPConv.add_module('DSBlock'+str(i), DownSamplingBlock(1, Channel))
            else:
                self.VPConv.add_module('DSBlock'+str(i), DownSamplingBlock(Channel, Channel))
                self.VMConv.add_module('DSBlock'+str(i-1), DownSamplingBlock(Channel, Channel))

        # Linear Projection
        self.FLinear = nn.ModuleList([nn.Sequential(nn.Linear(self.VecLen, self.BandVecLen), nn.LayerNorm(self.BandVecLen)) for i in range(self.Channel)])
        self.GLinear = nn.ModuleList([nn.Sequential(nn.Linear(self.VecLen, self.BandVecLen), nn.LayerNorm(self.BandVecLen)) for i in range(self.Channel)])
        # FineAdjust
        self.FineAdjust = nn.Conv2d(Channel, Channel, (3,3), 1, 1)
    def forward(self, guide, x):
        up_x = fun.interpolate(x, scale_factor=(4,4), mode='nearest')
        Fm = self.FMConv(up_x)
        Fq = self.FPConv(guide)
        F = self.FConv(torch.cat([Fm, Fq], dim=1))
        Gm = self.GMConv(x)
        Gp = self.GPConv(guide)
        G = self.GConv(torch.cat([Gm, Gp], dim=1))
        Vm = self.VMConv(x)
        Vp = self.VPConv(guide)
        V = self.VConv(torch.cat([Vm, Vp], dim=1))
        C = V.shape[1]
        batch = G.shape[0]
        W, H = F.shape[2], F.shape[3]
        OW, OH = G.shape[2], G.shape[3]
        G = torch.transpose(torch.transpose(G, 1, 2), 2, 3)
        G = G.reshape(batch*OW*OH, self.VecLen)
        F = torch.transpose(torch.transpose(F, 1, 2), 2, 3)
        F = F.reshape(batch*W*H, self.VecLen)
        BandsProbability = None
        for i in range(C):
            # F projection
            FVF = self.GLinear[i](G)
            FVF = FVF.reshape(batch, OW*OH, self.BandVecLen).transpose(-1, -2) # (batch, L, OW*OH)
            # G projection
            PVF = self.FLinear[i](F)
            PVF = PVF.view(batch, W*H, self.BandVecLen) # (batch, W*H, L)
            # Probability
            Probability = torch.bmm(PVF, FVF).reshape(batch*H*W, OW, OH) / sqrt(self.BandVecLen)
            Probability = torch.exp(Probability) / torch.sum(torch.exp(Probability), dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)
            Probability = Probability.view(batch, W, H, 1, OW, OH)
            # Merge
            if BandsProbability is None:
                BandsProbability = Probability
            else:
                BandsProbability = torch.cat([BandsProbability, Probability], dim=3)
        #Information Entropy: H_map = torch.sum(BandsProbability*torch.log2(BandsProbability+1e-9), dim=(-1, -2, -3)) / C
        out = torch.sum(BandsProbability*V.unsqueeze(dim=1).unsqueeze(dim=1), dim=(-1, -2))
        out = out.transpose(-1, -2).transpose(1, 2)
        out = self.FineAdjust(out)
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, kernel_num):
        super(ResidualBlock, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=in_channels, padding=1, kernel_size=kernel_size, out_channels=kernel_num)
        self.Conv2 = nn.Conv2d(in_channels=in_channels, padding=1, kernel_size=kernel_size, out_channels=kernel_num)
    def forward(self, x):
        y = F.relu(self.Conv1(x), False)
        y = self.Conv2(y)
        return x + y
      
class PanNet_PGCU(nn.Module):
    def __init__(self, channel, veclen, kernel_size=(3,3), kernel_num=32):
        super(PanNet_PGCU, self).__init__()
        self.PGCU = PGCU(hs_channels, veclen)
        self.Conv1 = nn.Conv2d(in_channels=channel+PAN_channels, padding=1, kernel_size=kernel_size, out_channels=kernel_num)
        self.ResidualBlocks = nn.Sequential(ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num))
        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=channel, padding=1, kernel_size=kernel_size)
    def forward(self, PAN, LRHS, hpan):
        HRHS = self.PGCU(PAN, LRHS)
        x = torch.cat([hpan, HRHS], dim=1)
        y = fun.relu(self.Conv1(x))
        y = self.ResidualBlocks(y)
        y = self.Conv2(y)
        return y + HRHS        

net = PanNet_PGCU(hs_channels,128).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
loss_fn = nn.MSELoss().to(device)
