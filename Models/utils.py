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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getMemory():
    allocated_memory = torch.cuda.memory_allocated()
    reserved_memory = torch.cuda.memory_reserved()

    print(f"Memory Allocated: {allocated_memory / (1024 ** 3):.2f} GB")
    print(f"Memory Reserved: {reserved_memory / (1024 ** 3):.2f} GB")

def MaxMargin(fMRI_embed, image_embed, batch_size, margin_value=2):

    criterion = nn.MSELoss()

    pos_loss = criterion(fMRI_embed, image_embed)

    repeat_image = image_embed.repeat(1,batch_size).reshape(batch_size,batch_size,256)
    repeat_image_T = repeat_image.transpose(0, 1)
    neg_loss =  criterion(repeat_image, repeat_image_T)

    margin = torch.tensor(margin_value) 
    total_loss = pos_loss + torch.max(torch.tensor(0.0), margin - neg_loss)

    return total_loss, pos_loss, neg_loss

def BasicCTR(fMRI_normal, image_normal, BATCH_SIZE, temperature=.125):
    # Already normalized
    logits = torch.mm(fMRI_normal, image_normal.T) # Positive samples are on the diagonal, negative off diagonal
    labels = torch.arange(BATCH_SIZE, device=device)
    row_loss = F.cross_entropy(logits / temperature, labels)
    return row_loss

def SoftCTR(fMRI_normal, image_normal, caption_normal, temperature=.1):

    # Already Normalized

    # Get Labels
    labels = torch.mm(caption_normal, caption_normal.T)

    # Get Probabilities
    logits = torch.mm(image_normal, caption_normal.T)

    # Loss
    f = nn.Softmax(dim=-1)
    fl = nn.LogSoftmax(dim=-1)

    loss1 = -(f(labels/temperature) * fl(logits/temperature)).sum(dim=-1).mean()
    loss2 = -(f(labels.T/temperature) * fl(logits/temperature)).sum(dim=-1).mean()
    loss_image_text = (loss1 + loss2)/2

    criterion = nn.MSELoss()  
    loss_fMRI_image = criterion(image_normal.detach(), fMRI_normal)

    return loss_image_text, loss_fMRI_image 