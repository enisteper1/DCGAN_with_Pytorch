from models import Generator
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import argparse
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from utils import tensor2image

def test_gen(args):
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        
    net_gen = Generator(args.nc, args.latent, args.img_size).to(device)
    
    net_gen.load_state_dict(torch.load(args.weight))
    
    noise = torch.randn(args.bsize, args.latent, 1, 1, device=device)
    output  = net_gen(noise).detach().cpu()
    
    for i in range(args.bsize):
        image = tensor2image(output[i])
        image.save(args.outdir + "/" + str(i) + ".jpg")
    
    print(f"{args.bsize} image created")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='weights/netG_last.pt', help='Generator weight path')
    parser.add_argument('--outdir', type=str, default='output', help='Output directory')
    parser.add_argument("--latent", type=int, default=256, help="size of generator input")
    parser.add_argument('--img_size', type=int, default=64, help='inference size (pixels)')
    parser.add_argument('--bsize', type=int, default=64, help='batchsize')
    parser.add_argument("--nc", type=int, default=3, help="number of channels")
    
    args = parser.parse_args()
    
    print(args)
    
    test_gen(args)