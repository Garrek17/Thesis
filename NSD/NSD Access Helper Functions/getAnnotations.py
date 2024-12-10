
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
    captions =  nsda.read_image_coco_info(indices)
    return captions


annotations_df = pd.DataFrame(getAnnot(list(range(73000)))).replace(r'\s+', ' ', regex=True)
annotations_df.to_csv('/home/gmc62/project/Thesis/Data/captions.csv', index=False, header=False)