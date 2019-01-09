import torch
import torch.nn as nn
import torch.nn.functional as F

class forward_conv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(forward_conv,self).__init__()
        self.conv=nn.Sequential( nn.Conv2d(in_ch,out_ch,3,stride=1,padding=1),
                                nn.BatchNorm2d(out_ch),
                                nn.LeakyReLU(0.2,inplace=True),
                                nn.Conv2d(out_ch,out_ch,3,stride=1,padding=1),
                                nn.BatchNorm2d(out_ch),
                                nn.LeakyReLU(0.2,inplace=True),
                                )
    def forward(self,x):
        x=self.conv(x)
        return x
    
class down(nn.Module):
    def __init__(self):
        super(down,self).__init__()
        self.pool=nn.Sequential(nn.MaxPool2d(kernel_size=2))
        
    def forward(self,x):
        x=self.pool(x)
        return x

class Up(nn.Module):
    def __init__(self,in_ch,out_ch,bilinear=False):
        super(Up,self).__init__()
        if bilinear:
            self.up=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up=nn.ConvTranspose2d(in_ch,out_ch,kernel_size=2,stride=2)
            
    def forward(self,xup,xconv):
        xup=self.up(xup)
        x=torch.cat([xconv,xup],dim=1)
        return x
        
            




class NightSight(nn.Module):
    def __init__(self):
        super(NightSight,self).__init__()
        
        self.conv1=forward_conv(4,32)
        self.pool1=down()
        
        self.conv2=forward_conv(32,64)
        self.pool2=down()
        
        self.conv3=forward_conv(64,128)
        self.pool3=down()
        
        self.conv4=forward_conv(128,256)
        self.pool4=down()
        
        self.conv5=forward_conv(256,512)
        
        
        self.up1=Up(512,256,bilinear=False)
        self.conv4_2=forward_conv(512,256)
        
        self.up2=Up(256,128,bilinear=False)
        self.conv3_2=forward_conv(256,128)
        
        self.up3=Up(128,64,bilinear=False)
        self.conv2_2=forward_conv(128,64)
        
        self.up4=Up(64,32,bilinear=False)
        self.conv1_2=forward_conv(64,32)
        
        self.conv10=nn.Conv2d(32,12,kernel_size=1,stride=1)
        
        
    def forward(self,x):
        conv1=self.conv1(x)
        pool1=self.pool1(conv1)
        
        conv2=self.conv2(pool1)
        pool2=self.pool2(conv2)
        
        conv3=self.conv3(pool2)
        pool3=self.pool3(conv3)
        
        conv4=self.conv4(pool3)
        pool4=self.pool4(conv4)
        
        conv5=self.conv5(pool4)
        
        up1=self.up1(conv5,conv4)
        conv4_2=self.conv4_2(up1)
        
        
        up2=self.up2(conv4_2,conv3)
        conv3_2=self.conv3_2(up2)
        
        
        up3=self.up3(conv3_2,conv2)
        conv2_2=self.conv2_2(up3)
        
        up4=self.up4(conv2_2,conv1)
        conv1_2=self.conv1_2(up4)
        
        conv10= self.conv10(conv1_2)
        
        out=nn.functional.pixel_shuffle(conv10,2)
        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
        
        
        