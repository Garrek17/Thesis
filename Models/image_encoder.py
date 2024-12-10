import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super(ImageEncoder, self).__init__()
        # Input Dim = 425 x 425
        self.encoder_stack = nn.Sequential(
            # Output Dim = [Input Dim - Kernel Size + 2*Padding]/stride + 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1), # [512, 212, 212]
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), # [256, 106, 106]
            nn.ReLU(),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), # [256, 53, 53]
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=4, padding=1), # [256, 13, 13]
            nn.ReLU(),
    
            nn.Flatten(),  
            nn.Linear(256 * 13 * 13, 256 * 7 * 7 ), 
            nn.ReLU(), 
            nn.Linear(256 * 7 * 7, embed_dim),  

        )

    def forward(self, x):
        x = self.encoder_stack(x)  
        x = F.normalize(x, p=2, dim=1)
        return x

class ImageDecoder(nn.Module):
    def __init__(self, embed_dim=256):
        super(ImageDecoder, self).__init__()
        self.decoder_stack = nn.Sequential(
            nn.Linear(embed_dim, 256 * 7 * 7),  
            nn.ReLU(),
            nn.Linear(256 * 7 * 7, 256 * 13 * 13),  
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(256, 13, 13)),  
            
            # output_size = (input_size−1)*stride−2*padding+kernel_size
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=6, stride=4, padding=1), # [batch, 26, 26]
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=6, stride=2, padding=1), # [batch, 53, 53]
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), # [batch, 212, 212]
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=5, stride=2, padding=1), # [batch, 425, 425]
            nn.Sigmoid()  
            
        )

    def forward(self, x):
        x = self.decoder_stack(x)  
        return x


# model = ImageEncoder(embed_dim=256).to(device)
# summary(model, input_size=(3, 425, 425)) 

# model = ImageDecoder(embed_dim=256).to(device)
# summary(model, input_size=(256,)) 