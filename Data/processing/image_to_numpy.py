# INPUT: Algonauts Data Directory
# OUTPUT: Numpy array of train and test directories of shape
# (#subjects, #images per subject, #channels, dim_width, dim_height)

import os
from PIL import Image
import numpy as np

def process_images(base_dir, subject, split):
    images_dir = os.path.join(base_dir, subject, f'{split}_split', f'{split}_images')
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])

    subject_images = []

    for image_file in image_files:
        print(image_file)
        img_path = os.path.join(images_dir, image_file)
        with Image.open(img_path) as img:
            img_rgb = img.convert('RGB')
            img_array = np.array(img_rgb)
            img_array = np.transpose(img_array, (2, 0, 1))  # (height, width, 3) -> (3, height, width)
            subject_images.append(img_array)

    subject_images_np = np.stack(subject_images)
    save_path = os.path.join(base_dir, subject, f'{split}_split', f'{split}_images.npy')
    np.save(save_path, subject_images_np)
    print(f"Saved NumPy {split.capitalize()} array for {subject} at: {save_path}")

def process_all_subjects(base_dir):
    # subjects = [subject for subject in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, subject))]
    subjects = ['subj01']
    for subject in subjects:
        print(f"Processing images for subject: {subject}")
        
        # Process training images
        process_images(base_dir, subject, 'training')

base_dir = '/home/gmc62/NSD'
process_all_subjects(base_dir)
