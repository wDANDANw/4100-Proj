import torch
import torch.nn as nn
import torch.optim as optim

# Imports for showing dataset
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from GlobalVariables import channels, feature_map_size, latent_size  # For class definitions, normal fields
from GlobalVariables import ngpu, beta1, num_epochs_encoder  # For encoder training
from GlobalVariables import device, dataloader


class Encoder(nn.Module):
    def __init__(self, ngpu):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(

            nn.Conv2d(channels, feature_map_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Conv2d(feature_map_size, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(feature_map_size * 2, feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Conv2d(feature_map_size * 4, feature_map_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(feature_map_size * 8, feature_map_size, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.LeakyReLU(0.2),
        )

        self.fc_class = nn.Sequential(
            nn.Linear(feature_map_size, latent_size),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.main(input)
        x = x.view(x.size(0), -1)
        x = self.fc_class(x)
        return x


class Decoder(nn.Module):
    def __init__(self, ngpu):
        super(Decoder, self).__init__()
        self.ngpu = ngpu

        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_size, feature_map_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_map_size * 2, feature_map_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_map_size, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, embedding):
        embedding = embedding.view(embedding.size(0), embedding.size(1), 1, 1)
        reconstructed_output = self.main(embedding)
        return reconstructed_output


class InnerAutoEncoder(nn.Module):
    def __init__(self, ngpu):
        super(InnerAutoEncoder, self).__init__()

        self.encoder = Encoder(ngpu).to(device)
        self.decoder = Decoder(ngpu).to(device)

    def forward(self, input, mode='train'):
        if mode == 'train':
            x = self.encoder(input)
            return self.decoder(x)

        if mode == 'generate':
            return self.decoder(input)


class AutoEncoder:
    # Defines the architecture of the encoder

    def __init__(self):

        assert not (device is None or dataloader is None)

        self.num_epochs = num_epochs_encoder

        self.auto_encoder = InnerAutoEncoder(ngpu)
        self.criterion = nn.MSELoss()

        self.eps = 1e-10
        self.lr = 1e-4
        self.optimizer = optim.Adam(self.auto_encoder.parameters(), lr=self.lr, betas=(beta1, 0.999))
        self.__trainAutoEncoder__(self.num_epochs, self.criterion, self.eps, self.lr)

    def __trainAutoEncoder__(self, num_epochs, criterion, eps, lr, print_training_report=False):
        """
        Train Auto Encoder
        Training Loop
        Initialize BCELoss function
        """

        # Lists to keep track of progress
        img_list = []
        iters = 0
        self.losses = []
        print("Starting Auto Encoder Training Loop...")
        # For each epoch
        for epoch in range(num_epochs):
            for i, data in enumerate(dataloader, 0):

                self.auto_encoder.zero_grad()
                image = data[0].to(device)
                b_size = image.size(0)

                reconstructed_image = self.auto_encoder(image).view(-1)

                err = criterion(reconstructed_image.view(image.size()) + eps, image)
                err.backward()

                self.optimizer.step()

                # Output training stats
                if i % 50 == 0:
                    print('[{0}/{1}][{2}/{3}] \t Loss: {4}'.format(epoch, num_epochs, i, len(dataloader), err.item()))

                # Save Losses for plotting later
                self.losses.append(err.item())

                # # Check how the generator is doing by saving G's output on fixed_noise
                # if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                #     with torch.no_grad():
                #         fake = auto_encoder(fixed_noise,data[1].long().to(device)).detach().cpu()
                #     img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1
            lr *= 0.995
            self.optimizer = optim.Adam(self.auto_encoder.parameters(), lr=lr, betas=(beta1, 0.999))

            if print_training_report:
                plt.figure(figsize=(10, 5))
                plt.title("AE Loss During Training")
                plt.plot(self.losses)
                plt.xlabel("iterations")
                plt.ylabel("Loss")
                plt.legend()
                plt.show()

                fake = []
                batch = 4
                # Grab a batch of real images from the dataloader
                real_batch = next(iter(dataloader))

                # Plot the real images
                plt.figure(figsize=(10, 10))
                plt.subplot(1, 2, 1)
                plt.axis("off")
                plt.title("Real Images")
                plt.imshow(
                    np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),
                                 (1, 2, 0)))

                with torch.no_grad():
                    for k in range(1):
                        fixed_noise = torch.randn(100, 3, 64, 64, device=device)
                        plt.figure(figsize=(10, 10))
                        # print(fixed_noise.size())
                        print(real_batch[0].size())
                        real_noised = real_batch[0].to(device) + fixed_noise * 0.9
                        plt.axis("off")
                        plt.title("Real Images + Noise")
                        plt.imshow(np.transpose(
                            vutils.make_grid(real_noised.to(device)[:64], padding=5, normalize=True).cpu(),
                            (1, 2, 0)))
                        fake = self.auto_encoder(real_noised.to(device))
                        # print(fake.size())
                        plt.figure(figsize=(10, 10))
                        plt.axis("off")
                        plt.title("Reconstructed Images")
                        plt.imshow(
                            np.transpose(vutils.make_grid(fake.to(device)[:64], padding=5, normalize=True).cpu(),
                                         (1, 2, 0)))

    def getAutoEncoder(self):
        return self.auto_encoder
