import torch.nn as nn
import torch
import torch.optim as optim

import matplotlib.pyplot as plt

from GlobalVariables import channels, feature_map_size, condition_size, latent_size, ngpu
from GlobalVariables import batch_size, beta1, num_epochs_generator
from GlobalVariables import device, dataloader

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
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

        )
        self.real_fake = nn.Sequential(
            nn.Conv2d(feature_map_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(feature_map_size * 8, feature_map_size, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.LeakyReLU(0.2),
        )

        self.fc_class = nn.Sequential(
            nn.Linear(feature_map_size, 4),
            nn.Sigmoid()
        )

    def forward(self, input, mode='train'):
        if mode == 'train':
            x = self.main(input)
            return self.real_fake(x)
        if mode == 'pretrain':
            x = self.main(input)
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            x = self.fc_class(x)
            return x


class InnerGenerator(nn.Module):
        def __init__(self, ngpu):
            super(InnerGenerator, self).__init__()
            self.ngpu = ngpu

            self.condition_embedding = nn.Embedding(4, condition_size)

            self.main = nn.Sequential(
                nn.ConvTranspose2d(latent_size + condition_size, feature_map_size * 8, 4, 1, 0, bias=False),
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

        def forward(self, input, condition):
            x = self.condition_embedding(condition)
            if len(x.size()) == 1:
                x = x.view(1, x.size(0), 1, 1)
            else:
                x = x.view(x.size(0), x.size(1), 1, 1)
            x = torch.cat((input, x), 1)
            return self.main(x)


class Generator:

    def __init__(self):
        assert not (device is None or dataloader is None)

        self.netD = Discriminator(ngpu).to(device)

        # Handle multi-gpu if desired
        if (device.type == 'cuda') and (ngpu > 1):
            self.netD = nn.DataParallel(self.netD, list(range(ngpu)))

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        self.netD.apply(weights_init)

        # Create the generator
        self.netG = InnerGenerator(ngpu).to(device)

        # Handle multi-gpu if desired
        if (device.type == 'cuda') and (ngpu > 1):
            self.netG = nn.DataParallel(self.netG, list(range(ngpu)))

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        self.netG.apply(weights_init)
        self.__trainDCGAN__()

    def __trainDCGAN__(self, print_training_report=False):

        # Training Loop

        # Initialize BCELoss function
        criterion_ce = nn.CrossEntropyLoss()
        criterion = nn.BCELoss()
        criterion_cos = nn.HingeEmbeddingLoss()

        def wasserstein_loss(y_true, y_pred):
            return torch.mean(y_true * y_pred)

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(batch_size, latent_size, 1, 1, device=device)

        # Establish convention for real and fake labels during training
        real_label = 1
        fake_label = 0

        lr_d = 0.00002
        lr_g = 0.00008
        # Setup Adam optimizers for both G and D
        optimizerD = optim.Adam(self.netD.parameters(), lr=lr_d, betas=(beta1, 0.999))
        optimizerG = optim.Adam(self.netG.parameters(), lr=lr_g, betas=(beta1, 0.999))

        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        print("Starting Generator Training Loop...")
        # For each epoch

        for epoch in range(num_epochs_generator):
            for i, data in enumerate(dataloader, 0):

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.netD.zero_grad()
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)

                # Soft label between 1.1 and 0.9
                label = ((1.1 - 0.8) * torch.rand((b_size, 1)) + 0.8).to(device)

                output = self.netD(real_cpu, 'train').view(-1,1)

                # Bug: ValueError: Using a target size (torch.Size([100, 1])) that is different to the input size
                # (torch.Size([100])) is deprecated. Please ensure they have the same size.
                # Solve: change output.view(-1) to output.view(-1, 1)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                noise = torch.randn(b_size, latent_size, 1, 1, device=device)
                fake = self.netG(noise, data[1].long().to(device))

                label = (0.1 * torch.rand((b_size, 1))).to(device)

                output = self.netD(fake.detach(), 'train').view(-1, 1)
                errD_fake = criterion(output, label)

                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake

                output = self.netD(real_cpu, 'pretrain')
                errD += criterion_ce(output, data[1].to(device)) * 0.5

                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                label.fill_(real_label)
                output = self.netD(fake, 'train').view(-1, 1)
                errG = criterion(output, label)

                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, num_epochs_generator, i, len(dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                iters += 1
        if print_training_report:
            plt.figure(figsize=(10, 5))
            plt.title("Generator and Discriminator Loss During Training")
            plt.plot(G_losses, label="G")
            plt.plot(D_losses, label="D")
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

    def printNetG(self):
        # Print the model
        print(self.netG)

    def printNetD(self):
        # Print the model
        print(self.netD)
