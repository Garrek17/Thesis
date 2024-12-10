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
from transformers import CLIPProcessor, CLIPModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


model_CLIP = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model_CLIP.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", do_rescale=False)

images = torch.from_numpy(np.load('/home/gmc62/NSD/subj01/training_split/training_images.npy')).to(device)

embed = torch.empty((0, 512)).to(device)
for i, image in enumerate(images):
    print(i)
    image = image / 255.0
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        image_CLIP = model_CLIP.get_image_features(inputs["pixel_values"])
    embed = torch.cat((embed, image_CLIP), dim=0).to(device)

torch.save(embed, '/home/gmc62/NSD/subj01/training_split/training_images_CLIP.pth')


