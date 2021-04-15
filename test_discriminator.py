import os
import cv2
import torch
import random
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
from models import Discriminator, Generator
import argparse
import cv2
from PIL import Image, ImageGrab
import glob
import time

def test_disc(args):
    os.makedirs("output", exist_ok=True)
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    data_transforms = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    net_disc = Discriminator(args.nc, args.image_size).to(device)
    net_disc.load_state_dict(torch.load(args.weight))

    with torch.no_grad():
        img_list = glob.glob(args.source + "*.jpg")
        im1 = data_transforms(Image.open(img_list[0])).reshape(1, args.nc, args.image_size, args.image_size)
        im2 = data_transforms(Image.open(img_list[1])).reshape(1, args.nc, args.image_size, args.image_size)
        images = torch.cat((im1, im2), dim=0)
        
        max_len = args.bsize if len(img_list) > args.bsize else len(img_list)
        for image_name in img_list[2:max_len]:
            image = data_transforms(Image.open(image_name)).reshape(1, args.nc, args.image_size, args.image_size)
            images = torch.cat((images, image), dim=0)
     
        
        for image_name in img_list:
            img = data_transforms(Image.open(image_name)).reshape(1, args.nc, args.image_size, args.image_size)
            images = images[:-1]
            images = torch.cat((images, img), dim=0)
            output = net_disc(images.to(device)).view(-1)
            print(image_name, end=" ")
            print("Real" if output[-1]>0.5 else "Fake") 
            time.sleep(0.01)   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='weights/netD_last.pt', help='Discriminator weight path')
    parser.add_argument('--source', type=str, default='source/', help='source')
    parser.add_argument('--image_size', type=int, default=64, help='inference size (pixels)')
    parser.add_argument('--bsize', type=int, default=64, help='batchsize')
    parser.add_argument("--nc", type=int, default=3, help="number of channels")
    args = parser.parse_args()
    print(args)
    test_disc(args)
    