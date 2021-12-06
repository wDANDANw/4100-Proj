import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch

# Imports for showing dataset
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

# Load Dataset

# Load dataset
# Dataset can be download here: [github.com/agamiko/pixel_character_generator](https://github.com/AgaMiko/pixel_character_generator/)
#
# from google.colab import files
# uploaded = files.upload()
# !unzip -q data.zip #uzipping dataset

dataroot = "."
show_initial_dataset = False


class DataLoader:

    def __init__(self, image_size=64, batch_size=100, workers=2, ngpu=1):
        # Create the dataset
        self.dataset = dset.ImageFolder(root=dataroot + '/data',
                                        transform=transforms.Compose([
                                            transforms.Resize(image_size),
                                            transforms.CenterCrop(image_size),
                                            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3,
                                                                   hue=0.1),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ]))
        # Create the dataloader
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                                      shuffle=True, num_workers=workers)

        # Decide which device we want to run on
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
        print("Using " + str(self.device))

    def getDataset(self):
        return self.dataset

    def getDataLoader(self):
        return self.dataloader

    def getDevice(self):
        return self.device

    # Plot some training images
    def ShowInitialDataset(self):
        real_batch = next(iter(self.dataloader))
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(
            np.transpose(vutils.make_grid(real_batch[0].to(self.device)[:64], padding=2, normalize=True).cpu(),
                         (1, 2, 0)))
        plt.show()
