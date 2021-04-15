import torch
import torch.optim as optim
import argparse
import os
from pathlib import Path
from models import *
from utils import *

def train(args):

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        
    # Defining Generator & Discriminator                                        
    net_gen  = Generator(nc=args.nc, latent=args.latent, image_size=args.image_size).to(device)
    net_disc = Discriminator(nc=args.nc, image_size=args.image_size).to(device)
    
    if args.resume:
        print("Loading weights...")
        net_gen.load_state_dict(torch.load(args.weightG))
        net_disc.load_state_dict(torch.load(args.weightD))
    
    else:
       print("Initialized weights...")
       net_gen.apply(weights_init)
       net_disc.apply(weights_init)
    
    if not Path("weights").exists():
        os.makedirs("weights")
    # Criterion to calculate loss   since its 1 or 0 binary cross entropy loss is used
    criterion = nn.BCELoss()
    
    # Define optimizers
    optimizer_disc = optim.Adam(net_disc.parameters(), lr = args.lr, betas = (args.beta, 0.999))
    optimizer_gen = optim.Adam(net_gen.parameters(), lr = args.lr, betas = (args.beta, 0.999))
    
    # Schedulers used to decrease learning rate at every ( epochs // 4 ). Every 75th epoch is defined as default
    scheduler_disc = optim.lr_scheduler.StepLR(optimizer_disc, step_size=args.epochs // 4, gamma=args.gamma)
    scheduler_gen = optim.lr_scheduler.StepLR(optimizer_gen, step_size=args.epochs // 4, gamma=args.gamma)
    
    fixed_noise = torch.randn(args.bsize, args.latent, 1, 1, device=device)

    # Labels of fake and real data
    real_label = 1.0
    fake_label = 0.0
    decaying_noise = 1.0
    
    G_losses = []
    D_losses = []
    minD_loss = 100
    minG_loss = 100
    
    
    for epoch in range(args.epochs):
        # In order to increase dataset size and variation random augmentation steps applied at every epoch
        dataloader = get_data(args.dataroot, args.image_size, args.bsize)

        for i, data in enumerate(dataloader):
            # Train Discriminator with real data
            net_disc.zero_grad()
            data_real = data[0].to(device)
            data_size = data_real.size(0)
            label = torch.full((data_size,), real_label, dtype=torch.float, device=device)
            disc_noise = ((torch.randn(data_size,args.nc ,args.image_size,args.image_size) + 0.0) * decaying_noise).to(device)
            output = net_disc(data_real + disc_noise).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            
            ## Train Discriminator with fake data
            noise = torch.randn(data_size, args.latent, 1, 1, device=device)
            data_fake = net_gen(noise)
            disc_noise = ((torch.randn(data_size,args.nc ,args.image_size,args.image_size) + 0.0) * decaying_noise).to(device)
            output = net_disc(data_fake.detach() + disc_noise).view(-1)
            label.fill_(fake_label)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizer_disc.step()
            
            # Train Generator
            net_gen.zero_grad()
            output = net_disc(data_fake).view(-1)
            label.fill_(real_label)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizer_gen.step()
            
            # Show progress
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, args.epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
        # Step schedulers
        scheduler_disc.step()
        scheduler_gen.step()

        if epoch % (args.epochs // 4) == 0 and epoch != 0:
          decaying_noise *= 0.5
        
        # Save checkpoints & save current outputs
        if epoch % args.savefreq == 0:
            torch.save(net_gen.state_dict(), f"weights/netG{epoch}.pt")
            torch.save(net_disc.state_dict(), f"weights/netD{epoch}.pt")
            
            output  = net_gen(fixed_noise).detach().cpu()
            batch2image(output, epoch)
            
        # Save best models
        if errD.item() < minD_loss:
            torch.save(net_disc.state_dict(), "weights/netD_best.pt")
            minD_loss = errD.item()
            
        if errG.item() < minG_loss:
            torch.save(net_gen.state_dict(), "weights/netG_best.pt")
            minG_loss = errG.item()
            
        torch.save(net_gen.state_dict(), f"weights/netG_last.pt")
        torch.save(net_disc.state_dict(), f"weights/netD_last.pt")
    # Visualize the whole training progress
    visualize(G_losses, D_losses, True)

 
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="data", help= "path to data")
    parser.add_argument('--weightG', type=str, default='weights/netG_best.pt', help='Generator weight path')
    parser.add_argument('--weightD', type=str, default='weights/netD_best.pt', help='Discriminator weight path')
    parser.add_argument("--latent", type=int, default=256, help="size of generator input")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="if resuming to training")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--epochs", type=int, default=300, help="epoch number")
    parser.add_argument("--savefreq", type=int, default=25, help= "Save checkpoints every <savefreq>") 
    parser.add_argument('--image_size', type=int, default=64, help='inference size pixels')
    parser.add_argument('--bsize', type=int, default=64, help='batchsize')
    parser.add_argument("--gamma", type=float, default=0.1, help="value that will multiply with lr every (<epochs> // 4)")
    parser.add_argument("--beta", type=float, default=0.5, help="beta value for adam optimizer")
    parser.add_argument("--nc", type=int, default=3, help="number of channels")
    
    args = parser.parse_args()
    
    print(args)
    train(args)
    
    
    