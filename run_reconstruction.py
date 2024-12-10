###############################################################################
#Import Statements 
###############################################################################
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import random
import statistics
import phate
import seaborn as sns
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from einops import rearrange
from Data.dataloader import NSDDataset, coco_categories
from Model.utils import MaxMargin, BasicCTR, SoftCTR, getMemory
from Model.fMRI_encoder import fMRIEncoder, fMRIDecoder
from Model.image_encoder import ImageDecoder, ImageEncoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###############################################################################
#Parameters
###############################################################################
TRAIN = False
FRESH = False

ALIGN_METHOD = 'SoftCTR' # MaxMargin, BasicCTR, SoftCTR, Word2Vec
TRAIN_METHOD = 'Withheld' # ALL (all objects), Withheld (leave out certain objects)
OBJECTS_WITHHELD = [24, 21, 1, 78] # Giraffe, Elephant, Bicycle, Oven

EPOCHS = 500
BATCH_SIZE = 128
LEARNING_RATE_fMRI = .0008
LEARNING_RATE_Image = .001

###############################################################################
# Print Configuration
###############################################################################
print("\nConfiguration Settings:")
print(f"    TRAIN: {TRAIN}")
print(f"    FRESH: {FRESH}")
print(f"    ALIGN_METHOD: {ALIGN_METHOD}")
print(f"    WITHHELD: {TRAIN_METHOD=='Withheld'}")
print(f"    OBJECTS_WITHHELD: {OBJECTS_WITHHELD}\n")

###############################################################################
# Load Model Paths
###############################################################################
if ALIGN_METHOD == 'MaxMargin':
    FILE_MODEL_fMRI_Enc = f"Model/Models/fMRI_Enc_MaxMargin_{TRAIN_METHOD}.pth"
    FILE_MODEL_Image_Enc = f"Model/Models/Image_Enc_MaxMargin_{TRAIN_METHOD}.pth"
    FILE_MODEL_fMRI_Dec = f"Model/Models/fMRI_Dec_MaxMargin_{TRAIN_METHOD}.pth"
    FILE_MODEL_Image_Dec = f"Model/Models/Image_Dec_MaxMargin_{TRAIN_METHOD}.pth"
    losses = {'total':[], 'pos':[], 'neg':[]}
    align_color = 'mistyrose'
    embed_dim = 256

elif ALIGN_METHOD == 'BasicCTR':
    FILE_MODEL_fMRI_Enc = f"Model/Models/fMRI_Enc_BasicCTR_{TRAIN_METHOD}.pth"
    FILE_MODEL_Image_Enc = f"Model/Models/Image_Enc_BasicCTR_{TRAIN_METHOD}.pth"
    FILE_MODEL_fMRI_Dec = f"Model/Models/fMRI_Dec_BasicCTR_{TRAIN_METHOD}.pth"
    FILE_MODEL_Image_Dec = f"Model/Models/Image_Dec_BasicCTR_{TRAIN_METHOD}.pth"
    embed_dim = 256

elif ALIGN_METHOD == 'SoftCTR':
    FILE_MODEL_fMRI_Enc = f"Model/Models/fMRI_Enc_SoftCTR_{TRAIN_METHOD}.pth"
    FILE_MODEL_Image_Enc = f"Model/Models/Image_Enc_SoftCTR_{TRAIN_METHOD}.pth"
    FILE_MODEL_fMRI_Dec = f"Model/Models/fMRI_Dec_SoftCTR_{TRAIN_METHOD}.pth"
    FILE_MODEL_Image_Dec = f"Model/Models/Image_Dec_SoftCTR_{TRAIN_METHOD}.pth"
    embed_dim = 300

elif ALIGN_METHOD == 'Word2Vec':
    FILE_MODEL_fMRI_Enc = f"Model/Models/fMRI_Enc_Word2Vec_{TRAIN_METHOD}_2.pth"
    FILE_MODEL_Image_Enc = f"Model/Models/Image_Enc_Word2Vec_{TRAIN_METHOD}_2.pth"
    FILE_MODEL_fMRI_Dec = f"Model/Models/fMRI_Dec_Word2Vec_{TRAIN_METHOD}_2.pth"
    FILE_MODEL_Image_Dec = f"Model/Models/Image_Dec_Word2Vec_{TRAIN_METHOD}_2.pth"
    embed_dim = 300

if TRAIN:
    dataset = NSDDataset('/home/gmc62/NSD', TRAIN_METHOD=='Withheld', OBJECTS_WITHHELD)
else:
    dataset = NSDDataset('/home/gmc62/NSD', False, OBJECTS_WITHHELD)

print(f"Dataset Length: {len(dataset)}\n")
torch.manual_seed(42)
train_size = int(0.8 * len(dataset)) 
test_size = len(dataset) - train_size  
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True, drop_last=True) 

###############################################################################
# Define Train Function
###############################################################################

def train(FRESH):

    checkpoint_fMRI = torch.load(FILE_MODEL_fMRI_Enc)
    fMRI_encoder = checkpoint_fMRI['model']
    checkpoint_Image = torch.load(FILE_MODEL_Image_Enc)
    image_encoder = checkpoint_Image['model']
    fMRI_encoder.eval()
    image_encoder.eval()

    # Whether to train from scratch or checkpoint
    if FRESH == True:
        fMRI_decoder = fMRIDecoder(embed_dim=embed_dim).to(device)
        image_decoder = ImageDecoder(embed_dim=embed_dim).to(device)
        ep_base = 0
           
    else:
        checkpoint_fMRI = torch.load(FILE_MODEL_fMRI_Dec)
        fMRI_decoder = checkpoint_fMRI['model']
        checkpoint_Image = torch.load(FILE_MODEL_Image_Dec)
        image_decoder = checkpoint_Image['model']
        ep_base = checkpoint_fMRI['epoch']

    criterion = nn.MSELoss()  
    fMRI_optimizer = optim.Adam(fMRI_decoder.parameters(), lr=LEARNING_RATE_fMRI)
    image_optimizer = optim.Adam(image_decoder.parameters(), lr=LEARNING_RATE_Image)

    fMRI_encoder.eval()
    image_encoder.eval()
    fMRI_decoder.train()
    image_decoder.train()

    for ep in range(EPOCHS):

        losses = {'total':[], 'fMRI':[], 'image':[]}

        for i, (caption_Text, caption_Word2Vec, category_OneHot, category_largest_Word2Vec, image, lh_fMRI, rh_fMRI) in enumerate(train_loader):
            
            fMRI_optimizer.zero_grad()
            image_optimizer.zero_grad()

            # fMRI Recon
            combined_fMRI = torch.cat((lh_fMRI, rh_fMRI), dim=1).unsqueeze(1)
            fMRI_recon = fMRI_decoder(fMRI_encoder(combined_fMRI))
            fMRI_loss = criterion(fMRI_recon, combined_fMRI)
            fMRI_loss.backward()
            losses['fMRI'].append(fMRI_loss.item())
            fMRI_optimizer.step()
            
            # Image Recon
            image = image / 255.0
            image_recon = image_decoder(image_encoder(image))
            image_loss = criterion(image, image_recon)
            image_loss.backward()
            losses['image'].append(image_loss.item())
            image_optimizer.step()

            total = fMRI_loss + image_loss
            losses['total'].append(total.item())

        print(f"EPOCH: {ep}")
        max_label_length = max(len(l) for l in losses.keys()) 
        for l in losses.keys():
            avg_loss = sum(losses[l]) / len(losses[l])
            print(f"  {l.ljust(max_label_length)}: {avg_loss:.5f}")

        if ep % 10 == 0 and ep != 0:
            print("Saving...")
            torch.save({'epoch': ep_base + ep, 'model': fMRI_decoder}, FILE_MODEL_fMRI_Dec)
            torch.save({'epoch': ep_base + ep, 'model': image_decoder}, FILE_MODEL_Image_Dec)
            print("Done Saving...")

###############################################################################
# Evaluate Alignment
###############################################################################
def evaluate():

    checkpoint_fMRI_Enc = torch.load(FILE_MODEL_fMRI_Enc)
    fMRI_encoder = checkpoint_fMRI_Enc['model']
    checkpoint_fMRI_Dec = torch.load(FILE_MODEL_fMRI_Dec)
    fMRI_decoder = checkpoint_fMRI_Dec['model']
    checkpoint_Image_Enc = torch.load(FILE_MODEL_Image_Enc)
    image_encoder = checkpoint_Image_Enc['model']
    checkpoint_Image_Dec = torch.load(FILE_MODEL_Image_Dec)
    image_decoder = checkpoint_Image_Dec['model']
    ep_base = checkpoint_fMRI_Dec['epoch']
    print(f'Starting on epoch: {ep_base}\n')

    fMRI_encoder.eval()
    image_encoder.eval()
    fMRI_decoder.eval()
    image_decoder.eval()


    for i, (caption_Text, caption_Word2Vec, category_OneHot, category_largest_Word2Vec, image, lh_fMRI, rh_fMRI) in enumerate(test_loader):
        
        target_indices = np.where(category_OneHot[:,21]==1)[0:10]

        # Reconstruction of Image from Image
        image_normalized = image[target_indices] / 255.0
        image_from_image = image_decoder(image_encoder(image_normalized))

        # Reconstruction of Image from fMRI
        combined_fMRI = torch.cat((lh_fMRI[target_indices], rh_fMRI[target_indices]), dim=1).unsqueeze(1)
        image_from_fMRI = image_decoder(fMRI_encoder(combined_fMRI))

        fig, axs = plt.subplots(3, 6, figsize=(18, 9))  
        for i in range(6):
            
            random_input_image = image_normalized[i].detach().cpu().permute(1, 2, 0).numpy()
            random_image_from_image = image_from_image[i].detach().cpu().permute(1, 2, 0).numpy()
            random_image_from_fMRI = image_from_fMRI[i].detach().cpu().permute(1, 2, 0).numpy()
            
            axs[0, i].imshow(random_input_image)
            axs[0, i].axis('off')
            
            axs[1, i].imshow(random_image_from_image)
            axs[1, i].axis('off')
            
            axs[2, i].imshow(random_image_from_fMRI)
            axs[2, i].axis('off')

        plt.tight_layout()
        plt.savefig(f'combined_image_grid_{ALIGN_METHOD}.png', dpi=300)
        plt.show()

        return

if __name__ == '__main__':
    if TRAIN == True:
        train(FRESH)
    else:
        evaluate()