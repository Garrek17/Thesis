
import os
import re
import glob
import urllib.request
import zipfile
import pandas as pd
import heapq
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from einops import rearrange
import nibabel as nb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from nilearn import plotting
from pycocotools.coco import COCO
from nsd_access import NSDAccess


def getAnnot(indices):
    fmri_path = '/gpfs/milgram/data/nsd/'
    nsda = NSDAccess(fmri_path)
    annot =  nsda.read_image_coco_info(indices)
    return annot


coco_categories = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 
    71: 'oven', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 
    76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 
    80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 
    86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 
    90: 'toothbrush'
}


annots = getAnnot(list(range(73000)))

all_images = []
weights = []


for i, image in enumerate(annots):
    print(i)
    dict = {}
    for segment in image:
        if segment['category_id'] not in dict.keys():
            dict[segment['category_id']] = 0
        dict[segment['category_id']] += segment['area']

    top_three_keys = heapq.nlargest(3, dict, key=dict.get)
    top_three_areas = [dict[key] for key in top_three_keys]
    all_images.append([coco_categories[i] for i in top_three_keys])

    total_area = sum(top_three_areas)
    weights.append([area / total_area for area in top_three_areas])


weights_df = pd.DataFrame(weights)
weights_df.to_csv('/home/gmc62/project/Thesis/Data/largest_categories_weights.csv', index=False, header=False)

annotations_df = pd.DataFrame(all_images)
annotations_df.to_csv('/home/gmc62/project/Thesis/Data/largest_categories.csv', index=False, header=False)