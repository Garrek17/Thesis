
import os
import re
import glob
import urllib.request
import zipfile
import pandas as pd
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
    return nsda.read_image_coco_category(indices)

annotations_df = pd.DataFrame(getAnnot(list(range(73000)))) 
annotations_df.to_csv('/home/gmc62/project/Thesis/Data/categories.csv', index=False, header=False)