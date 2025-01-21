### TMAR: 3D Transformer Network via Masked Autoencoder Regularization for Hyperspectral Sharpening ###
### doi:  ###

def upsample(x, h, w):
    f= F.interpolate(x, size=[h,w], mode='bicubic', align_corners=False)
    return f
    
class MyConvTranspose2d(nn.Module):
    def __init__(self, conv, output_size):
        super(MyConvTranspose2d, self).__init__()
        self.output_size = output_size
        self.conv = conv
    def forward(self, x):
        x = self.conv(x, output_size=self.output_size).to(device)
        return x
    
class SE_comp(nn.Module):
    def __init__(self, hs_channels):
        super(SE_comp, self).__init__()
        self.conv1=nn.Conv1d(1,1,1,1) 
        self.lin1=nn.Linear(hs_channels,hs_channels)
        self.lin2=nn.Linear(hs_channels,hs_channels)
        self.avgP1=torch.nn.AdaptiveAvgPool2d((1,1))
        self.avgP2 = torch.nn.AdaptiveAvgPool1d(1)
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        x1  = self.avgP1(x).squeeze(dim=0).permute(1,2,0)
        k   = self.lin1(x1)
        q   = self.lin2(x1)
        x2  = self.avgP2(k.permute(0,2,1)@q)
        x3  = self.sigmoid( self.conv1(x2.permute(1,2,0)).unsqueeze(dim=0))
        out = x * x3
        return out
SE = SE_comp(hs_channels) 

class ViTModel(nn.Module):
    def __init__(self, hs_channels):
        super(ViTModel, self).__init__()
        mid_channels = int(hs_channels/2)
        depth_num = 5
        self.conv3d1 = nn.Conv3d(hs_channels, mid_channels, (5,5,depth_num) , padding=(2,2,2)).to(device)
        self.conv3d2 = nn.Conv3d(mid_channels, mid_channels, (5,5,depth_num) , padding=(2,2,2)).to(device)
        self.conv3d3 = nn.Conv3d(mid_channels, hs_channels, (5,5,depth_num) , padding=(2,2,2)).to(device)
        self.conv3d4 = nn.Conv3d(hs_channels, hs_channels, (5,5,depth_num) , padding=(2,2,2)).to(device)
        self.conv3d5 = nn.Conv3d(hs_channels, hs_channels, (5,5,depth_num) , padding=(2,2,2)).to(device)
        self.conv3d6 = nn.Conv3d(mid_channels, hs_channels, (5,5,depth_num) , padding=(2,2,2)).to(device)
        self.avgP1   = torch.nn.AdaptiveAvgPool1d(1)
        self.avgP2   = torch.nn.AdaptiveAvgPool2d((mid_channels,1))   
        self.sigmoid      = nn.Sigmoid()
    def selfAttention3d(self, s):
            k = (self.conv3d4(s))
            q = (self.conv3d5(s))
            v = s
            k = k.squeeze(dim=0)
            k = k.squeeze(dim=3).permute(1,2,0)
            q = q.squeeze(dim=0)
            q = q.squeeze(dim=3).permute(1,2,0)        
            k = self.avgP1(k)
            q = self.avgP2(q)
            d = k@q.permute(0,2,1)
            d = d.permute(2,0,1)
            d = d.unsqueeze(dim=0)
            d = d.unsqueeze(dim=4)
            d = self.conv3d6(d)
            d = self.sigmoid(d)
            out = ((d/2 + v))
            return out
    def multiHeadAttention(self, x):
      s1    = self.selfAttention3d(x)
      s2    = self.selfAttention3d(x)
      s3    = self.selfAttention3d(x)
      out   = (s1 + s2 + s3) /3
      return out
    def forward(self, x):
        x = x.unsqueeze(4)
        mx = self.multiHeadAttention(x) + x
        self.norm1 = nn.LayerNorm([1,hs_channels, mx.shape[2], mx.shape[2],1]).to(device)
        mx = self.norm1(mx)
        #MLP
        e = (self.conv3d1(mx))
        e = (self.conv3d2(e))
        e = F.relu(self.conv3d3(e))
        e = (e + mx)/2
        e = self.norm1(e)
        out = (e.squeeze(4))        
        return out
ViT = ViTModel(hs_channels)      

class updateBlock(nn.Module):
    def __init__(self, hs_channels, PAN_channels):
        super(updateBlock, self).__init__()
        mid_channels = int(hs_channels/2)
        self.conv1    = nn.Conv2d(hs_channels, mid_channels, 1, padding=0).to(device)
        self.conv2    = nn.Conv2d(mid_channels, PAN_channels, 1, padding=0).to(device)
        self.conv3    = nn.Conv2d(PAN_channels, mid_channels, 1, padding=0).to(device)
        self.conv4    = nn.Conv2d(mid_channels, hs_channels, 1, padding=0).to(device)
        self.conv5    = nn.Conv2d(hs_channels, mid_channels, 1, padding=0).to(device)
        self.conv6    = nn.Conv2d(mid_channels, hs_channels, 1, padding=0).to(device)
        self.conv7    = nn.Conv2d(hs_channels, mid_channels, 3, padding=1).to(device)
        self.conv8    = nn.Conv2d(mid_channels, hs_channels, 3, padding=1).to(device)
        self.conv9    = nn.Conv2d(hs_channels, mid_channels, 3, padding=1).to(device)
        self.conv10    = nn.Conv2d(mid_channels, hs_channels, 3, padding=1).to(device)
        self.conv11    = nn.Conv2d(hs_channels, mid_channels, 3, padding=1).to(device)
        self.conv12    = nn.Conv2d(mid_channels, hs_channels, 3, padding=1).to(device)
    def spectralUpdate(self, x, LRHS):
        _,_,M,N = x.shape
        _,_,m,n = LRHS.shape
        x1 = upsample(self.conv8(F.relu(self.conv7(x))), m, n)       
        xdiff = LRHS - x1
        x2 = upsample(self.conv10(F.relu(self.conv9(xdiff))), M, N)
        out = self.conv12(F.relu(self.conv11(x + x2)))
        return out    
    def SpatialUpdate(self, x, PAN):
        PANr =  self.conv2(F.relu(self.conv1(x)))
        PANrT = PAN - PANr
        PANrT2 =  self.conv4(F.relu(self.conv3(PANrT))) 
        out =  self.conv6(F.relu(self.conv5(x + PANrT2)))           
        return out        
    def forward(self, x, LRHS, PAN):
        xU = self.spectralUpdate(x, LRHS)
        out = self.SpatialUpdate(xU, PAN)
        return out
updateBk = updateBlock(hs_channels, PAN_channels) 

class Maskautoencoder(nn.Module):
    def __init__(self, hs_channels, PAN_channels):
        super(Maskautoencoder, self).__init__()
        mid_channels = int(hs_channels/2)
        self.conv1    = nn.Conv2d(hs_channels, mid_channels, 1, padding=0).to(device)
        self.conv2    = nn.Conv2d(mid_channels, PAN_channels, 1, padding=0).to(device)
        self.conv3    = nn.Conv2d(PAN_channels, mid_channels, 1, padding=0).to(device)
        self.conv4    = nn.Conv2d(mid_channels, mid_channels, 1, padding=0).to(device)
        self.conv5    = nn.Conv2d(hs_channels, mid_channels, 1, padding=0).to(device)
        self.conv6    = nn.Conv2d(mid_channels, hs_channels, 1, padding=0).to(device)
        self.conv7    = nn.Conv2d(hs_channels, mid_channels, 3, padding=1).to(device)
        self.conv8    = nn.Conv2d(mid_channels, hs_channels, 3, padding=1).to(device)
        self.conv9    = nn.Conv2d(hs_channels, mid_channels, 3, padding=1).to(device)
        self.conv10    = nn.Conv2d(mid_channels, hs_channels, 3, padding=1).to(device)
        self.conv11    = nn.Conv2d(hs_channels, mid_channels, 3, padding=1).to(device)
        self.conv12    = nn.Conv2d(mid_channels, hs_channels, 3, padding=1).to(device)
        self.dropout = nn.Dropout2d(p=0.2)
    def mEncoder(self, u,LRHS,PAN,loop):
        #percent = 25/100 # Number of unmasking channels
        #randlayer = random.sample(range(0,hs_channels),int(hs_channels*percent))
        #xen = u[:,randlayer,:,:]
        xen = u
        for d1 in range(loop):
            x_ViT   = ViT(xen) + xen
            xen2   = self.dropout(x_ViT)
            x_SE  = SE(xen2) + xen2
            xen3 = updateBk(x_SE, LRHS, PAN)  
        convs = nn.Conv2d(xen3.shape[1], hs_channels, 3, padding=1).to(device)  
        out = (convs(xen3)) 
        out = (out + xen)/2
        return out    
    def umDecoder(self, xe,LRHS,PAN):
        x_ViT   = ViT(xe) + xe
        xen2   = self.dropout(x_ViT)
        x_SE  = SE(xen2) + xen2
        out = updateBk(x_SE, LRHS, PAN)  
        out =  (xe + out)/2        
        return out        
    def forward(self, u,LRHS,PAN,loop):
        xe = self.mEncoder(u,LRHS,PAN,loop)
        out = self.umDecoder(xe,LRHS,PAN)
        return out
autoencoder = Maskautoencoder(hs_channels, PAN_channels) 

class TMAR(nn.Module):
    def __init__(self, hs_channels, PAN_channels):
        super(TMAR, self).__init__()
        self.deconv= nn.ConvTranspose2d(hs_channels, hs_channels, 3, padding=1, stride=4).to(device)
        self.conv1    = nn.Conv2d(hs_channels+PAN_channels, hs_channels, 3, padding=1).to(device)
        self.dropout = nn.Dropout2d(p=0.2)
           
    def forward(self, LRHS, PAN, loop): #LRHS = [1,hs_channels,width,height], PAN = [1,PAN_channels,width,height]    
        #deconv1 = MyConvTranspose2d(self.deconv, output_size=(PAN.shape[2], PAN.shape[3]))
        #HRHS = deconv1(LRHS)      
        HRHS = upsample(LRHS, PAN.shape[2], PAN.shape[3])
        x = torch.cat((HRHS, PAN), dim=1)
        x = self.conv1(x) + HRHS
        for i in range(loop):
            for i in range(loop):
                # CTU
                x_ViT   = ViT(x) + x
                x1   = self.dropout(x_ViT)
                x_SE  = SE(x1) + x1
                u = updateBk(x_SE, LRHS, PAN) 
                x =  (u + x)/2                    
            x = autoencoder(u,LRHS,PAN,loop)    
        x = x  + HRHS   
        return x    
net = TMAR(hs_channels, PAN_channels)
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=0)      

def loopCreator(train_error_rate):
  loop = math.ceil(math.log10(math.pow(train_error_rate, math.e-1)) + 3) + 1    
  if loop <= 0: loop = 1;
  return loop

def lossFn(x,y):      
        a = (x**2)+(y**2)
        b =  torch.abs(y-x)
        c = torch.sum(torch.mul(x,y))
        d = torch.sum(torch.mul(a,b))
        e = torch.div(d,c)
        f = torch.norm(x)*torch.norm(y)
        g = torch.div(c,f)
        h = torch.acos(g)
        if torch.isnan(h):
            loss = e;
        elif torch.isinf(h):
            loss = e;
        else :
            loss = e*h
        return loss
  
