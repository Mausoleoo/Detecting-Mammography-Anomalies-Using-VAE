import torch
from torch import nn


class VAEConv(nn.Module):
    def __init__(self, input_size=128, image_channels = 1,  latent_dim=100, Show_Error = False):
        super().__init__()
        self.input_size = input_size
        
        aux = int(256 * (input_size // 16) * (input_size // 16))
        # Encoder Conv
        #Example input size 128 * 128
        self.enc1 = nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=3, stride=1, padding=1) #32*64*64 (After Maxpooling)
        self.enc2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) #64*32*32 (After Maxpooling)
        self.enc3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) #128*16*16 (After Maxpooling)
        self.enc4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1) #256*8*8 (After Maxpooling)

        
        # Latent Space

        self.mu = nn.Linear(aux, latent_dim) 
        self.logvar = nn.Linear(aux , latent_dim)

        # Decoder Cov
        self.fc_z = nn.Linear(latent_dim, aux) 

        self.dec4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1) #256*8*8 (After Deconv)
        self.dec3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1) #128*16*16 (After Deconv)
        self.dec2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1) #64*32*32 (After Deconv)
        self.dec1 = nn.ConvTranspose2d(in_channels=32, out_channels=image_channels, kernel_size=3, stride=2, padding=1, output_padding=1) #32*64*64 (After Deconv)
              
        self.sigmoid = nn.Sigmoid()

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

        self.reludec4 = nn.ReLU()
        self.reludec3 = nn.ReLU()
        self.reludec2 = nn.ReLU()
        self.reludec1 = nn.ReLU()

        self.pooling1 = nn.MaxPool2d(2, stride=2)
        self.pooling2 = nn.MaxPool2d(2, stride=2)
        self.pooling3 = nn.MaxPool2d(2, stride=2)
        self.pooling4 = nn.MaxPool2d(2, stride=2)


    def Encode(self, x):
        x = self.enc1(x)
        x = self.relu1(x)
        x = self.pooling1(x)

        x = self.enc2(x)
        x = self.relu2(x)
        x = self.pooling2(x)

        x = self.enc3(x)
        x = self.relu3(x)
        x = self.pooling3(x)

        x = self.enc4(x)
        x = self.relu4(x)
        x = self.pooling4(x)

        x = x.view(x.size(0), -1)
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

    def Decode(self, x, input_size, apply_sigmoid=False):
        x = self.fc_z(x)
        x = x.view(-1, 256, int(input_size//16), int(input_size//16))

        x = self.dec4(x)
        x = self.reludec4(x)

        x = self.dec3(x)
        x = self.reludec3(x)

        x = self.dec2(x)
        x = self.reludec2(x)

        x = self.dec1(x)
        #x = self.reludec1(x)

        #if apply_sigmoid == True:
        x = self.sigmoid(x)

        return x
    
    def ELBO(self, x, x_hat, mu, logvar, Show_Error = False): #Evidence Lower Bound
        x_hat = torch.clamp(x_hat, 1e-5, 1. - 1e-5)
        reconstruction_loss = nn.functional.binary_cross_entropy(x, x_hat, reduction='sum')
        KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) #Kullback-Leibler Divergence
        error = reconstruction_loss + KLD * 0.1
        if Show_Error == True:
            print("Reconstruction Loss: " + str(reconstruction_loss) + "| KLD: " + str(KLD))
        return error

    def forward(self, x, input_size):
        x = self.Encode(x)
        mu, logvar = self.Latent_Space(x)
        z = self.Reparameterization(mu, logvar)
        x_hat = self.Decode(z, input_size)
        return x_hat, mu, logvar
