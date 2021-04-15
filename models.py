import torch
import torch.nn.functional as F
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Discriminator(nn.Module):
    def __init__(self, nc, image_size):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=nc, out_channels=image_size, kernel_size=4, stride=2, padding=1, bias=False)

        self.conv2 = nn.Conv2d(in_channels=image_size, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnrm2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.dropout3 = nn.Dropout(0.2)
        self.bnrm3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        self.dropout4 = nn.Dropout(0.2)
        self.bnrm4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False)


    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.3, inplace=True)
        x = F.leaky_relu(self.bnrm2(self.conv2(x)), negative_slope=0.3, inplace=True)
        x = F.leaky_relu(self.bnrm3(self.dropout3(self.conv3(x))), negative_slope=0.3, inplace=True)
        x = F.leaky_relu(self.bnrm4(self.dropout4(self.conv4(x))), negative_slope=0.3, inplace=True)
        
        return (torch.sigmoid(self.conv5(x)))
        

class Generator(nn.Module):
    def __init__(self, nc, latent, image_size):
        super(Generator, self).__init__()
        self.convt1 = nn.ConvTranspose2d(in_channels=latent, out_channels=512, kernel_size=4, stride=1, padding=0, bias=False)
        self.bnrm1 = nn.BatchNorm2d(512)
        
        self.convt2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnrm2 = nn.BatchNorm2d(256)

        self.convt3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnrm3 = nn.BatchNorm2d(128)
        
        self.convt4 = nn.ConvTranspose2d(in_channels=128, out_channels=image_size, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnrm4 = nn.BatchNorm2d(image_size)

        self.convt5 = nn.ConvTranspose2d(in_channels=image_size, out_channels=nc, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.bnrm1(self.convt1(x)), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.bnrm2(self.convt2(x)), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.bnrm3(self.convt3(x)), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.bnrm4(self.convt4(x)), negative_slope=0.2, inplace=True)
        
        return torch.tanh(self.convt5(x))