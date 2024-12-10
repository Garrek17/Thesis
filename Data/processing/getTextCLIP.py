from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import random
import torch
import csv
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from transformers import AutoTokenizer, CLIPTextModelWithProjection

model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
csv_reader = csv.reader(open("/home/gmc62/NSD/captions.csv", "r"))


if __name__ == "__main__":
    embed = torch.empty((0, 512)).to(device)
    for i, text in enumerate(csv_reader):
        print(i)
        with torch.no_grad():
            inputs = tokenizer(text, padding=True, return_tensors="pt").to(device)
            outputs = model(**inputs)
            text_embeds = outputs.text_embeds
            embed = torch.cat((embed, text_embeds), dim=0).to(device)

    torch.save(embed, "/home/gmc62/NSD/caption_CLIP.pth")

