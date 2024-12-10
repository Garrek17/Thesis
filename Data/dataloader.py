from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
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

# For now we train on 4 subjects
# Note that the dataset has a train test split, 
# but we ignore the test because they don't have corresponding fMRI per Algonauts Challenge
# Leave out subjects 6 and 8 because they have different dimensions 

class NSDDataset(Dataset):
    def __init__(self, NSD_path, WITHHELD, OBJECTS_WITHHELD):

        self.NSD_path = NSD_path
        self.WITHHELD = WITHHELD
        self.OBJECTS_WITHHELD = OBJECTS_WITHHELD
        self.device = device
        self.subject = 'subj01' # , 'subj02', 'subj03', 'subj04', 'subj05', 'subj07'

        self.captions_Text = np.loadtxt('/home/gmc62/NSD/captions.csv', delimiter='\t', dtype=str) 
        self.captions_Word2Vec = torch.load(os.path.join(NSD_path, 'captions_Word2Vec.pth'))
        self.categories_OneHot = np.load('/home/gmc62/NSD/categories_one_hot.npy')
        self.categories_largest_Word2Vec = torch.load('/home/gmc62/NSD/largest_categories_embed.pth')
        self.legal_indices = self.getIndices(self.subject)

        print(f"Loading {self.subject} into memory...")
        self.images = np.load(os.path.join(self.NSD_path, self.subject, 'training_split', 'training_images.npy'))
        self.subject_length = len(self.legal_indices)
        self.lh_fMRI = np.load(os.path.join(self.NSD_path, self.subject, 'training_split', 'training_fmri', 'lh_training_fmri.npy'))
        self.rh_fMRI = np.load(os.path.join(self.NSD_path, self.subject, 'training_split', 'training_fmri', 'rh_training_fmri.npy'))
        print("Done with dataloader init")

    def getIndices(self, subject):
        # To map back to one of the 73K NSD images, we look at the image file name
        indices = []
        images_dir = os.path.join(self.NSD_path, subject, 'training_split', 'training_images')
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
        for i, image_file in enumerate(image_files):
            nsd_id = int(image_file.split('nsd-')[-1].replace('.png', ''))
            # Remove caption indices in OBJECTS_WITHHELD if WITHHELD == True
            if self.WITHHELD and any(self.categories_OneHot[nsd_id][i] == 1 for i in self.OBJECTS_WITHHELD):
                continue
            indices.append((i, nsd_id))
        return indices

    def __len__(self):
        # Per Algonauts Challenge, right now length is just subject 1 length
        return len(self.legal_indices)

    def __getitem__(self, idx):

        random_idx = random.choice(self.legal_indices)
        ID_1 = random_idx[0] # to index into images or fMRI scans
        ID_2 = random_idx[1] # to index into captions or categories

        image = torch.from_numpy(self.images[ID_1]).float().to(self.device) 
        lh_fMRI = torch.from_numpy(self.lh_fMRI[ID_1]).float().to(self.device)
        rh_fMRI = torch.from_numpy(self.rh_fMRI[ID_1]).float().to(self.device)

        caption_Text =  self.captions_Text[ID_2]
        caption_Word2Vec = self.captions_Word2Vec[ID_2]
        category_OneHot = self.categories_OneHot[ID_2]
        category_largest_Word2Vec = self.categories_largest_Word2Vec[ID_2]

        return caption_Text, caption_Word2Vec, category_OneHot, category_largest_Word2Vec, image, lh_fMRI, rh_fMRI
        
coco_categories = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 
    10: 'fire hydrant', 12: 'stop sign', 13: 'parking meter', 14: 'bench', 
    15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep', 
    20: 'cow', 21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe', 
    26: 'backpack', 27: 'umbrella', 30: 'handbag', 31: 'tie', 32: 'suitcase', 
    33: 'frisbee', 34: 'skis', 35: 'snowboard', 36: 'sports ball', 37: 'kite', 
    38: 'baseball bat', 39: 'baseball glove', 40: 'skateboard', 41: 'surfboard', 
    42: 'tennis racket', 43: 'bottle', 45: 'wine glass', 46: 'cup', 47: 'fork', 
    48: 'knife', 49: 'spoon', 50: 'bowl', 51: 'banana', 52: 'apple', 
    53: 'sandwich', 54: 'orange', 55: 'broccoli', 56: 'carrot', 57: 'hot dog', 
    58: 'pizza', 59: 'donut', 60: 'cake', 61: 'chair', 62: 'couch', 
    63: 'potted plant', 64: 'bed', 66: 'dining table', 69: 'toilet', 
    71: 'tv', 72: 'laptop', 73: 'mouse', 74: 'remote', 75: 'keyboard', 
    76: 'cell phone', 77: 'microwave', 78: 'oven', 79: 'toaster', 
    80: 'sink', 81: 'refrigerator', 83: 'book', 84: 'clock', 85: 'vase', 
    86: 'scissors', 87: 'teddy bear', 88: 'hair drier', 89: 'toothbrush'
}

# # Test a Pair
# image, caption_text, caption_CLIP, caption, category, largest_categories_embed, lh_fMRI, rh_fMRI = dataset[10]
# plt.imsave('input.png', image.detach().cpu().permute(1, 2, 0).numpy().astype("uint8"))
# print(caption_text)
# indices =  list(np.where(category==1)[0])
# print([coco_categories[i] for i in indices])