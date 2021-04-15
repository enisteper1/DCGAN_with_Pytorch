import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

def get_data(dataroot="data", image_size=64, bsize=64):
    dataset = datasets.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),
                                   transforms.RandomAutocontrast(),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
                                   
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bsize,
                                             shuffle=True, num_workers=0)
                                             
    return dataloader


def tensor2image(tensor):
    data_transforms = transforms.ToPILImage()
    image = vutils.make_grid(tensor, padding=2, normalize=True)
    image = data_transforms(image)
    image = image.convert('RGB')
    return image
    

def batch2image(batch, epoch):
    batch = vutils.make_grid(batch, padding=2, normalize=True)
    image = np.transpose(batch,(1,2,0))
    plt.figure(figsize=(15,15))
    plt.axis("off")
    plt.imshow(image)
    plt.savefig(f"output/epoch{epoch}.png")
    print(f"Current output is saved to output directory as epoch{epoch}.png")
    
    
def visualize(G_losses, D_losses, savefig=True):
    plt.figure(figsize=(15,10))
    plt.title("Generator and Discriminator Loss")
    plt.plot(G_losses,label="Generator", color="black")
    plt.plot(D_losses,label="Discriminator", color="blue")
    plt.xlabel("Iteration number")
    plt.ylabel("Loss value")
    plt.legend()
    plt.grid()
    if savefig:
        plt.savefig("progress.png")
    plt.show()