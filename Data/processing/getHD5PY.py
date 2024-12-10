import os
import h5py
from PIL import Image
import numpy as np

# Directory containing the NSD data
base_dir = '/home/gmc62/NSD'

# Path to the output HDF5 file
output_hdf5_file = '/home/gmc62/NSD/nsd_train_images.h5'

# Create or open the HDF5 file
with h5py.File(output_hdf5_file, 'w') as hdf5_file:
    
    # Get list of subjects (directories in base_dir)
    subjects = [subject for subject in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, subject))]
    
    for subject in subjects:
        print(f"Processing training images for subject: {subject}")
        
        # Directory where training images are stored for this subject
        images_dir = os.path.join(base_dir, subject, 'training_split', 'training_images')
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])

        subject_images = []
        
        # Iterate through image files, convert them to NumPy arrays
        for image_file in image_files:
            img_path = os.path.join(images_dir, image_file)
            with Image.open(img_path) as img:
                img_rgb = img.convert('RGB')
                img_array = np.array(img_rgb)
                img_array = np.transpose(img_array, (2, 0, 1))  # (height, width, 3) -> (3, height, width)
                subject_images.append(img_array)

        # Convert the list of images into a NumPy array
        subject_images_np = np.stack(subject_images)

        # Create or access the subject dataset in the HDF5 file
        dataset_path = f"{subject}"
        hdf5_file.create_dataset(dataset_path, data=subject_images_np, compression="gzip", chunks=True)
        print(f"Saved training images for {subject} in HDF5 dataset at {dataset_path}")
