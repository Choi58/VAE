import torch
from torch import nn
from torchsummary import summary
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class enc_block(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(input_dim,input_dim,3,padding=1),
            nn.BatchNorm2d(input_dim),
            nn.LeakyReLU(),
            nn.Conv2d(input_dim,output_dim,3,stride=2,padding=1),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU()
        )
        self.identity = nn.Sequential(
            nn.Conv2d(input_dim,output_dim,3,stride=2,padding=1),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.identity(x) + self.res(x)
        
class encoder(nn.Module):
    def __init__(self):
        super().__init__()
        latent_dim = 128
        dims = [32,64,128,256,512]
        self.enc_input = nn.Sequential(
            nn.Conv2d(3,dims[0],3,stride=2,padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU()
            )
        enc = [enc_block(dims[i],dims[i+1]) for i in range(len(dims)-1)]
        self.enc = nn.Sequential(*enc)
        self.mu = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*dims[-1],latent_dim))
        self.sig = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*dims[-1],latent_dim))
    def forward(self,x):
        y = self.enc_input(x)
        y = self.enc(y)
        mu = self.mu(y)
        sig = self.sig(y)
        sig = torch.exp(sig) ##수정
        eps = torch.randn_like(sig)
        z = mu + sig*eps
        return [mu,sig,z]
class dec_block(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(input_dim,input_dim,3,padding=1),
            nn.BatchNorm2d(input_dim),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(input_dim,output_dim,3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU()
        )
        self.identity = nn.Sequential(
            nn.ConvTranspose2d(input_dim,output_dim,3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.identity(x) + self.res(x)
        
        
class decoder(nn.Module):
    def __init__(self):
        super().__init__()
        latent_dim = 128
        dims = [32,64,128,256,512]
        dims.reverse()
        self.dec_input = nn.Linear(latent_dim,4*dims[0])
        dec = [ dec_block(dims[i],dims[i+1]) for i in range(len(dims)-1)]
        self.dec = nn.Sequential(*dec)
        self.dec_output = nn.Sequential(
            nn.ConvTranspose2d(dims[-1],dims[-1],3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(dims[-1],3,3,padding=1),
            nn.Tanh()
        )
    def forward(self,x):
        y = self.dec_input(x)
        y = y.view(-1,512,2,2)
        y = self.dec(y)
        y = self.dec_output(y)
        return y

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = encoder()
        self.decoder = decoder()
    def forward(self,x):
        [mu,sig,z] = self.encoder(x)
        x_approx = self.decoder(z)
        return x_approx,x,mu,sig

#if __name__ == "__main__":
from torchviz import make_dot
model = VAE().to(device)
x = torch.rand(1,3,64,64).to(device)
[x_approx,x,mu,sig] = model(x)
make_dot(x_approx.mean(), params=dict(model.named_parameters()))