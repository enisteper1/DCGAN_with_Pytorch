## Introduction
Main purpose of this project for me to understand basics of how neural network and optimizers should be selected at DCGAN. While trying different datasets from kaggle
I tried optimizer SGD with nesterov instead of Adam. I was thinking about it may decrease the speed of training in contrast output quality of GAN would be increased. However, even if it decreased the training speed outputs were not better than Adam. Therefore, I changed generator model activation functions to leaky relu instead of relu. Also, added noise to discriminator and dropouts with the aim of preventing it to overwhelm generator. As a result, new pokemon images are created


## Train
```bash
$ python train.py
Namespace(beta=0.5, bsize=64, dataroot='data', epochs=100, gamma=0.1, image_size=64, latent=256, lr=0.01, nc=3, resume=False, savefreq=10, weightD='weights/netD_best.pt', weightG='weights/netG_best.pt')
Initialized weights...
[0/100][0/115]	Loss_D: 1.5639	Loss_G: 22.5150	D(x): 0.5503	D(G(z)): 0.4998 / 0.0000
[0/100][50/115]	Loss_D: 5.1343	Loss_G: 4.9279	D(x): 0.6908	D(G(z)): 0.5346 / 0.3185
[0/100][100/115]	Loss_D: 6.4388	Loss_G: 3.1877	D(x): 0.0152	D(G(z)): 0.0048 / 0.0806
Current output is saved to output directory as epoch0.png
[1/100][0/115]	Loss_D: 1.8535	Loss_G: 2.5551	D(x): 0.6679	D(G(z)): 0.3463 / 0.2009
[1/100][50/115]	Loss_D: 2.0337	Loss_G: 1.2507	D(x): 0.5177	D(G(z)): 0.5047 / 0.3824
```
 
## Improvement
<img src=https://user-images.githubusercontent.com/45767042/114794280-c9207400-9d94-11eb-9eca-b031da7b902b.gif>
<img src=https://user-images.githubusercontent.com/45767042/114804803-92edef00-9daa-11eb-83e2-ec51cb97c97e.png>

  
## Test
In order to get seperated outputs whenever the user wants I added `test_generator.py` which generates image with respect to batch size and saves individually. In addition,   `test_discriminator.py` that prints about image fake or real.

   ```bash
   $ python test_generator.py --weight weights/netG_last.pt
   Namespace(bsize=64, img_size=64, latent=256, nc=3, outdir='output', weight='weights/netG_last.pt')
   64 image created
   ```
   ```bash
   $ python test_discriminator.py
   Namespace(bsize=64, image_size=64, nc=3, source='source/', weight='weights/netD_last.pt')
   source\60.jpg Fake
   source\63.jpg Fake
   source\Pkmn_img1.jpg Real
   source\Pkmn_img3.jpg Real
   ```
   

## Reminder 
Images which are taken from kaggle  were ending with `.JPG`, in order to convert them to `.jpg` I wrote little script named as `converter.py`.
If you are going to use it you need to open zip file inside this folder. Otherwise, changing input directory is needed inside `converter.py`.

## Reference
- This project is mainly referenced from [DCGAN-Pytorch](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- Dataset is taken from [kaggle-pokemon](https://www.kaggle.com/djilax/pkmn-image-dataset)
