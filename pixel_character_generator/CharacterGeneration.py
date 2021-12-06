from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import random

from DataLoader import DataLoader

import GlobalVariables as GV
from GlobalVariables import latent_size


def Generate():
    # Set random seed for reproducibility
    manualSeed = '123'
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    data_loader = DataLoader(image_size=GV.image_size, batch_size=GV.batch_size, workers=GV.workers, ngpu=GV.ngpu)
    GV.device = data_loader.getDevice()
    GV.dataloader = data_loader.getDataLoader() # Used a lot in file, and therefore does not want to change name

    from Encoders import AutoEncoder
    from Trainer import Generator

    # Auto Encoder
    auto_encoder = AutoEncoder().getAutoEncoder()

    # Generator
    generator = Generator()

    fake = []
    batch = 100
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(GV.dataloader))

    # Plot the real images
    plt.figure(figsize=(25, 25))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(GV.device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))
    plt.show()

    with torch.no_grad():
        for i in range(1):
            fixed_noise = torch.randn(batch, latent_size, 1, 1, device=GV.device)
            plt.figure(figsize=(12, 12))
            # print(fixed_noise.size())
            print(real_batch[0].size())
            fake = generator.netG(fixed_noise, torch.tensor([i] * batch).to(GV.device)).detach().cpu()
            plt.axis("off")
            plt.title("Fake Images")
            plt.imshow(np.transpose(vutils.make_grid(fake.to(GV.device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))
            plt.show()
            fake = auto_encoder(fake.to(GV.device))
            # print(fake.size())
            plt.figure(figsize=(12, 12))
            plt.axis("off")
            plt.title("Fake and Reconstructed Images")
            plt.imshow(np.transpose(vutils.make_grid(fake.to(GV.device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))
            plt.show()

if __name__ == '__main__':
#     # Bug: recursive creating new processes => real_batch = next(iter(self.dataloader)), should be the dataloader, launches a server
#     # https://stackoverflow.com/questions/69515321/an-attempt-has-been-made-to-start-a-new-process-before-the-current-process-has-f
    Generate()