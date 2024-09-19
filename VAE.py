import torch
from torch import nn


class VAEConv(nn.Module):
    def __init__(self, image_channels = 1, init_channels = 8,  latent_dim=100, Show_Error = False):
        super().__init__()
        # Encoder Conv

        self.enc1 = nn.Conv2d(in_channels=image_channels, out_channels=init_channels, kernel_size=3, stride=1, padding='same')
        self.enc2 = nn.Conv2d(in_channels=init_channels, out_channels=init_channels*2, kernel_size=3, stride=1, padding='same')
        self.enc3 = nn.Conv2d(in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=3, stride=1, padding='same')
        self.enc4 = nn.Conv2d(in_channels=init_channels*4, out_channels=init_channels*8, kernel_size=3, stride=1, padding='same')

        
        # Latent Space

        self.mu = nn.Linear(init_channels * 8, latent_dim)
        self.logvar = nn.Linear(init_channels * 8, latent_dim)

        # Decoder Cov
        self.fc_z = nn.Linear(latent_dim, init_channels * 8)
        self.dec1 = nn.ConvTranspose2d(in_channels=init_channels * 8, out_channels=init_channels*4, kernel_size=3, stride=2, padding=0, output_padding=1)
        self.dec2 = nn.ConvTranspose2d(in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=3, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(in_channels=init_channels*2, out_channels=init_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec4 = nn.ConvTranspose2d(in_channels=init_channels, out_channels=image_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(2, stride=2)

    def Encode(self, x):
        print('Input')
        print(x.shape)
        x = self.enc1(x)
        x = self.relu(x)
        print('Conv 1')
        print(x.shape)
        x = self.pooling(x)
        print('Pooling 1')
        print(x.shape)
        x = self.enc2(x)
        x = self.relu(x)
        print('Conv 2')
        print(x.shape)
        x = self.pooling(x)
        print('Pooling 2')
        print(x.shape)
        x = self.enc3(x)
        x = self.relu(x)
        print('Conv 3')
        print(x.shape)
        x = self.pooling(x)
        print('Pooling 3')
        print(x.shape)
        x = self.enc4(x)
        x = self.relu(x)
        print('Conv 4')
        print(x.shape)
        x = self.pooling(x)
        print('Pooling 4')
        print(x.shape)
        x = torch.flatten(x, start_dim=1)
        return x
    
    def Latent_Space(self, x):
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def Reparameterization(self, mu, logvar):
        var = torch.exp(logvar * 0.5)
        epsilon = torch.randn_like(var)
        z = mu + epsilon * var
        return z

    def Decode(self, x):
        
        x = self.fc_z(x)
        x = x.view(-1, 64, 1, 1)
        x = self.dec1(x)
        x = self.relu(x)
        x = self.dec2(x)
        x = self.relu(x)
        x = self.dec3(x)
        x = self.relu(x)
        x = self.dec4(x)
        return torch.sigmoid(x)
    
    def ELBO(self, x, x_hat, mu, logvar, Show_Error = False): #Evidence Lower Bound
        x_hat = torch.clamp(x_hat, 1e-5, 1. - 1e-5)
        reconstruction_loss = nn.functional.binary_cross_entropy(x, x_hat, reduction = 'sum')
        KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) #Kullback-Leibler Divergence
        error = reconstruction_loss + KLD
        if Show_Error == True:
            print("Reconstruction Loss: " + str(reconstruction_loss) + "| KLD: " + str(KLD))
        return error

    def forward(self, x):
        x = self.Encode(x)
        mu, logvar = self.Latent_Space(x)
        z = self.Reparameterization(mu, logvar)
        x_hat = self.Decode(z)
        return x_hat, mu, logvar


class VAELinear(nn.Module):
    def __init__(self, input_dim = 28*28, hidden_dim = 200,  latent_dim=100, Show_Error = False):
        super().__init__()

        # Encoder Lineal

        self.enc_2hid1 = nn.Linear(input_dim, 512)
        self.hid1_2hid2 = nn.Linear(512, 256)
        self.hid2_2hid3 = nn.Linear(256, hidden_dim)

        # Latent Space

        self.hid_2mu = nn.Linear(hidden_dim, latent_dim)
        self.hid_2logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder Cov
        
        self.z_2hid3 = nn.Linear(latent_dim, hidden_dim)
        self.hid3_2hid2 = nn.Linear(hidden_dim, 256)
        self.hid2_2hid1 = nn.Linear(256, 512)
        self.hid1_2img = nn.Linear(512, input_dim)

        self.relu = nn.ReLU()
        
    def Encode(self, x):
        x = self.enc_2hid1(x)
        x = self.relu(x)
        x = self.hid1_2hid2(x)
        x = self.relu(x)
        x = self.hid2_2hid3(x)
        x = self.relu(x)
        return x
    
    def Latent_Space(self, x):
        mu = self.hid_2mu(x)
        logvar = self.hid_2logvar(x)
        return mu, logvar

    def Reparameterization(self, mu, logvar):
        var = torch.exp(logvar * 0.5)
        epsilon = torch.randn_like(var)
        z = mu + epsilon * var
        return z

    def Decode(self, x):
        x = self.z_2hid3(x)
        x = self.relu(x)
        x = self.hid3_2hid2(x)
        x = self.relu(x)
        x = self.hid2_2hid1(x)
        x = self.relu(x)
        x = self.hid1_2img(x)
        return torch.sigmoid(x)
    
    def ELBO(self, x, x_hat, mu, logvar, Show_Error): #Evidence Lower Bound
        x_hat = torch.clamp(x_hat, 1e-5, 1. - 1e-5)
        reconstruction_loss = nn.functional.binary_cross_entropy(x, x_hat, reduction = 'sum')
        KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) #Kullback-Leibler Divergence
        error = reconstruction_loss + KLD
        if Show_Error == True:
            print("Reconstruction Loss: " + str(reconstruction_loss) + "| KLD: " + str(KLD))
        return error

    def forward(self, x):
        x = self.Encode(x)
        mu, logvar = self.Latent_Space(x)
        z = self.Reparameterization(mu, logvar)
        x_hat = self.Decode(z)
        return x_hat, mu, logvar
    
