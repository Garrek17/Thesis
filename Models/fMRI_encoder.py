import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class fMRIEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super(fMRIEncoder, self).__init__()
        self.encoder_stack = nn.Sequential(
            # Input Dim = 39548
            # Output Dim = [Input Dim - Kernel Size + 2*Padding]/stride + 1
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=4, stride=8, padding=2), 
            nn.GELU(),
            
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=4, stride=4, padding=2), 
            nn.GELU(),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, stride=4, padding=2), 
            nn.GELU(),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=4, padding=2), 
            nn.GELU(),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=4, padding=2), 
            nn.GELU(),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=2, stride=4, padding=2), 
            nn.GELU(),

            # Additional layers for more depth
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=2, stride=2, padding=1),
            nn.GELU(),

            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=2, stride=2, padding=1),
            nn.GELU(),
            
            nn.Flatten(),
            # Adding more linear layers for a gradual transition
            nn.Linear(1024 * 3, 512*3),
            nn.GELU(),
            nn.Linear(512*3, 256*3),
            nn.GELU(),
            nn.Linear(256*3, embed_dim),
        )

    def forward(self, x):
        x = self.encoder_stack(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class fMRIDecoder(nn.Module):
    def __init__(self, embed_dim=256):
        super(fMRIDecoder, self).__init__()
        self.decoder_stack = nn.Sequential(
            # Adding more linear layers for a gradual transition
            # output_size = (input_size−1)*stride−2*padding+kernel_size
            nn.Linear(embed_dim, 256*3),
            nn.GELU(),
            nn.Linear(256*3, 512*3),
            nn.GELU(),
            nn.Linear(512*3, 1024 * 3),
            nn.GELU(),
            
            nn.Unflatten(dim=1, unflattened_size=(1024, 3)),
            
            nn.ConvTranspose1d(in_channels=1024, out_channels=512, kernel_size=4, stride=1, padding=1),
            nn.GELU(),

            nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=5, stride=3, padding=2),
            nn.GELU(),

            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=6, stride=4, padding=1),
            nn.GELU(),

            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=6, stride=4, padding=3),
            nn.GELU(),

            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=5, stride=4, padding=3),
            nn.GELU(),

            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=6, stride=4, padding=3),
            nn.GELU(),

            nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=8, stride=4, padding=2),
            nn.GELU(),

            nn.ConvTranspose1d(in_channels=8, out_channels=1, kernel_size=6, stride=4, padding=3),
        )


    def forward(self, x):
        x = self.decoder_stack(x)
        return x
