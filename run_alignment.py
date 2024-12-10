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

ALIGN_METHOD = 'Word2Vec' # MaxMargin, BasicCTR, SoftCTR, Word2Vec
TRAIN_METHOD = 'Withheld' # ALL (all objects), Withheld (leave out certain objects)
OBJECTS_WITHHELD = [24, 21, 1, 78] # Giraffe, Elephant, Bicycle, Oven

EPOCHS = 500 
BATCH_SIZE = 256
LEARNING_RATE_fMRI = .001
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
    losses = {'total':[], 'pos':[], 'neg':[]}
    align_color = 'mistyrose'
    embed_dim = 256

elif ALIGN_METHOD == 'BasicCTR':
    FILE_MODEL_fMRI_Enc = f"Model/Models/fMRI_Enc_BasicCTR_{TRAIN_METHOD}.pth"
    FILE_MODEL_Image_Enc = f"Model/Models/Image_Enc_BasicCTR_{TRAIN_METHOD}.pth"
    losses = {'total':[]}
    align_color = 'thistle'
    embed_dim = 256

elif ALIGN_METHOD == 'SoftCTR':
    FILE_MODEL_fMRI_Enc = f"Model/Models/fMRI_Enc_SoftCTR_{TRAIN_METHOD}.pth"
    FILE_MODEL_Image_Enc = f"Model/Models/Image_Enc_SoftCTR_{TRAIN_METHOD}.pth"
    losses = {'total':[], 'image-text':[], 'fMRI-image':[]}
    align_color = 'moccasin'
    embed_dim = 300

elif ALIGN_METHOD == 'Word2Vec':
    FILE_MODEL_fMRI_Enc = f"Model/Models/fMRI_Enc_Word2Vec_{TRAIN_METHOD}_2.pth"
    FILE_MODEL_Image_Enc = f"Model/Models/Image_Enc_Word2Vec_{TRAIN_METHOD}_2.pth"
    losses = {'fMRI-text':[], 'image-text':[]}
    align_color = 'mediumseagreen'
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
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=True) 

###############################################################################
# Define Train Function
###############################################################################
def train(FRESH):

    # Whether to train from scratch or checkpoint
    if FRESH == True:
        fMRI_encoder = fMRIEncoder(embed_dim=embed_dim).to(device)
        image_encoder = ImageEncoder(embed_dim=embed_dim).to(device)
        ep_base = 0
           
    else:
        checkpoint_fMRI = torch.load(FILE_MODEL_fMRI_Enc)
        fMRI_encoder = checkpoint_fMRI['model']
        checkpoint_Image = torch.load(FILE_MODEL_Image_Enc)
        image_encoder = checkpoint_Image['model']
        ep_base = checkpoint_fMRI['epoch']

    print(f'Starting on epoch: {ep_base}\n')
    criterion = nn.MSELoss()  
    optimizer = optim.Adam([
        {'params': fMRI_encoder.parameters(), 'lr': LEARNING_RATE_fMRI},
        {'params': image_encoder.parameters(), 'lr': LEARNING_RATE_Image}
    ])

    fMRI_encoder.train()
    image_encoder.train()
  
    for ep in range(EPOCHS):

        for i, (caption_Text, caption_Word2Vec, category_OneHot, category_largest_Word2Vec, image, lh_fMRI, rh_fMRI) in enumerate(train_loader):
                    
            optimizer.zero_grad()

            # fMRI 
            combined_fMRI = torch.cat((lh_fMRI, rh_fMRI), dim=1).unsqueeze(1)
            fMRI_embed = fMRI_encoder(combined_fMRI)

            # Image
            image = image / 255.0
            image_embed = image_encoder(image)

            # Normalize Captions
            caption_Word2Vec = F.normalize(caption_Word2Vec, p=2, dim=1)
            category_largest_Word2Vec = F.normalize(category_largest_Word2Vec, p=2, dim=1)
            combined_caption_embedding = .8*caption_Word2Vec + .2*category_largest_Word2Vec
            combined_caption_embedding = F.normalize(combined_caption_embedding, p=2, dim=1)


            if ALIGN_METHOD == 'MaxMargin':
                loss, pos, neg = MaxMargin(fMRI_embed, image_embed, BATCH_SIZE)
                losses['total'].append(loss.item())
                losses['pos'].append(pos.item())
                losses['neg'].append(neg.item())
                loss.backward()
                optimizer.step()
            
            elif ALIGN_METHOD == 'BasicCTR':
                loss = BasicCTR(fMRI_embed, image_embed, BATCH_SIZE)
                losses['total'].append(loss.item())
                loss.backward()
                optimizer.step()

            elif ALIGN_METHOD == 'SoftCTR':
                loss_image_text, loss_fMRI_image  = SoftCTR(fMRI_embed, image_embed, category_largest_Word2Vec)
                loss_image_text.backward()
                loss_fMRI_image.backward()
                loss = loss_image_text + loss_fMRI_image
                losses['image-text'].append(loss_image_text.item())
                losses['fMRI-image'].append(loss_fMRI_image.item())
                losses['total'].append(loss.item())
                optimizer.step()

            elif ALIGN_METHOD == 'Word2Vec':
                # fMRI - Caption Distance
                loss_fMRI = 1 - F.cosine_similarity(fMRI_embed, combined_caption_embedding.detach()).mean()
                loss_fMRI.backward()
                losses['fMRI-text'].append(loss_fMRI.item())

                # fMRI - Image Distance
                loss_image = 1 - F.cosine_similarity(image_embed, combined_caption_embedding.detach()).mean()
                loss_image.backward()
                losses['image-text'].append(loss_image.item())

                optimizer.step()

        print(f"EPOCH: {ep}")
        max_label_length = max(len(l) for l in losses.keys()) 
        for l in losses.keys():
            avg_loss = sum(losses[l]) / len(losses[l])
            print(f"  {l.ljust(max_label_length)}: {avg_loss:.5f}")

        if ep % 10 == 0 and ep != 0:
            print("Saving...")
            torch.save({'epoch': ep_base + ep, 'model': fMRI_encoder}, FILE_MODEL_fMRI_Enc)
            torch.save({'epoch': ep_base + ep, 'model': image_encoder}, FILE_MODEL_Image_Enc)
            print("Done Saving...")

    return

###############################################################################
# Evaluate Alignment
###############################################################################
def evaluate():

    checkpoint_fMRI = torch.load(FILE_MODEL_fMRI_Enc)
    fMRI_encoder = checkpoint_fMRI['model']
    checkpoint_Image = torch.load(FILE_MODEL_Image_Enc)
    image_encoder = checkpoint_Image['model']
    ep_base = checkpoint_fMRI['epoch']
    print(f'Starting on epoch: {ep_base}\n')
    fMRI_encoder.eval()
    image_encoder.eval()
    
    # Run Mini-Batch Due to Memory Constraint
    fMRI_embeddings = []
    image_embeddings = []
    categories_OneHot = []

    # Get Embeddings
    for i, (caption_Text, caption_Word2Vec, category_OneHot, category_largest_Word2Vec, image, lh_fMRI, rh_fMRI) in enumerate(test_loader):

        # fMRI
        combined_fMRI = torch.cat((lh_fMRI, rh_fMRI), dim=1).unsqueeze(1)
        fMRI_embed = fMRI_encoder(combined_fMRI)
        fMRI_embeddings.extend(fMRI_embed.detach().cpu().squeeze().tolist())

        # Image
        image = image / 255.0
        image_embed = image_encoder(image)
        image_embeddings.extend(image_embed.detach().cpu().squeeze().tolist())

        # Append the categories
        categories_OneHot.append(category_OneHot)

    categories_OneHot = torch.cat(categories_OneHot, dim=0).numpy()
    N = len(image_embeddings)

    # ####################################################################################
    # Get the PHATE Embeddings
    total_emb = fMRI_embeddings + image_embeddings
    total_emb_pandas = pd.DataFrame(total_emb)
    phate_operator = phate.PHATE(n_components=2, verbose=False)

    # Clustering to remove outliers
    digits_phate = phate_operator.fit_transform(total_emb_pandas)
    clusters = phate.cluster.kmeans(phate_operator, k=3)
    half_cluster = clusters[0:int(len(clusters)/2)]
    main_cluster = statistics.mode(clusters)

    # Get the statistics for the samples
    stats = np.sum(categories_OneHot, axis=0)

    # Color by specific categories
    categories_num = [22, 4, 58] # bear, toilet, motorcycle
    categories_value = [coco_categories[x] for x in categories_num]

    # Get the indices corresponding to each category
    indices = []
    for n in categories_num:
        indices.append(np.where((categories_OneHot[:,n]==1) & (half_cluster==main_cluster)))

    print(f"Top 3 Categories: {categories_value}")
    print("Red, Green, Blue")
    print(f"Frequency in Each Top Category: {[stats[i] for i in categories_num]}")

    phate_fMRI = digits_phate[0:N]
    phate_image = digits_phate[N:]

    class_1_fMRI = phate_fMRI[indices[0]]
    class_1_image = phate_image[indices[0]]

    class_2_fMRI = phate_fMRI[indices[1]]
    class_2_image = phate_image[indices[1]]

    class_3_fMRI = phate_fMRI[indices[2]]
    class_3_image = phate_image[indices[2]]

    remaining_fMRI = phate_fMRI[half_cluster==main_cluster]
    remaining_image = phate_image[half_cluster==main_cluster]

    # Color by Category
    plt.figure(figsize=(16, 12))

    # Plot of Non Class Objects
    plt.scatter(remaining_fMRI[:, 0], remaining_fMRI[:, 1], alpha=1,
                c='lightgrey', s=10, marker='o')  

    plt.scatter(remaining_image[:, 0], remaining_image[:, 1], alpha=.5,
                c='lightgrey', s=20, marker='*')  

    # Plot of Class 1 Objects
    plt.scatter(class_1_fMRI[:, 0], class_1_fMRI[:, 1], alpha=1,
                c='red', s=10, marker='o', label=f"{categories_value[0]} - fMRI")  

    plt.scatter(class_1_image[:, 0], class_1_image[:, 1], alpha=.5,
                c='red', s=20, marker='*', label=f"{categories_value[0]} - Image")  

    # Plot of Class 2 Objects
    plt.scatter(class_2_fMRI[:, 0], class_2_fMRI[:, 1], alpha=1,
                c='green', s=10, marker='o', label=f"{categories_value[1]} - fMRI")  

    plt.scatter(class_2_image[:, 0], class_2_image[:, 1], alpha=.5,
                c='green', s=20, marker='*', label=f"{categories_value[1]} - Image") 

    # Plot of Class 3 Objects
    plt.scatter(class_3_fMRI[:, 0], class_3_fMRI[:, 1], alpha=1,
                c='blue', s=10, marker='o', label=f"{categories_value[2]} - fMRI")  

    plt.scatter(class_3_image[:, 0], class_3_image[:, 1], alpha=.5,
                c='blue', s=20, marker='*', label=f"{categories_value[2]} - Image")  

    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])  
    ax.set_axis_off() 
    if ALIGN_METHOD == 'BasicCTR':
        ax.legend(fontsize=25, loc='upper right', markerscale=3, frameon=True, edgecolor='black', title=None )

    plt.savefig(f'phate_category_{ALIGN_METHOD}.png', bbox_inches='tight')  
    plt.close() 

    ####################################################################################
    # Histogram Similarity
    stats = {}

    # Statistics for Objects
    stats = {}
    object_indices = []
    for object in OBJECTS_WITHHELD:
        indices = np.where(categories_OneHot[:, object] == 1)[0]
        object_indices.extend(indices)

        # Euclidean Distance
        euclidean_distances = np.linalg.norm(
            np.array(image_embeddings)[indices] - np.array(fMRI_embeddings)[indices], axis=1
        ).flatten()
        euclidean_mean = np.mean(euclidean_distances)
        euclidean_std = np.std(euclidean_distances)

        # Cosine Similarity
        cosine_similarities = np.sum(
            np.array(image_embeddings)[indices] * np.array(fMRI_embeddings)[indices], axis=1
        ).flatten()
        cosine_mean = np.mean(cosine_similarities)
        cosine_std = np.std(cosine_similarities)

        stats[object] = {
            'Euclidean': {'Mean': euclidean_mean, 'Std': euclidean_std},
            'Cosine': {'Mean': cosine_mean, 'Std': cosine_std},
        }

    max_label_length = max(len(coco_categories[object].capitalize()) for object in stats.keys())
    for object, stats_obj in stats.items():
        print(
            f"{coco_categories[object].capitalize().ljust(max_label_length)}: "
            f"Euclidean Mean = {stats_obj['Euclidean']['Mean']:.2f}, "
            f"Euclidean Std = {stats_obj['Euclidean']['Std']:.2f}, "
            f"Cosine Mean = {stats_obj['Cosine']['Mean']:.2f}, "
            f"Cosine Std = {stats_obj['Cosine']['Std']:.2f}"
        )

    # Statistics for Remaining Categories
    all_indices = np.arange(N)
    remaining_indices = np.setdiff1d(all_indices, object_indices)

    # Euclidean Distance
    euclidean_distances = np.linalg.norm(
        np.array(image_embeddings)[remaining_indices] - np.array(fMRI_embeddings)[remaining_indices], axis=1
    ).flatten()
    euclidean_mean = np.mean(euclidean_distances)
    euclidean_std = np.std(euclidean_distances)

    # Cosine Similarity
    cosine_similarities = np.sum(
        np.array(image_embeddings)[remaining_indices] * np.array(fMRI_embeddings)[remaining_indices], axis=1
    ).flatten()
    cosine_mean = np.mean(cosine_similarities)
    cosine_std = np.std(cosine_similarities)

    print(
        f"{'All'.ljust(max_label_length)}: "
        f"Euclidean Mean = {euclidean_mean:.2f}, "
        f"Euclidean Std = {euclidean_std:.2f}, "
        f"Cosine Mean = {cosine_mean:.2f}, "
        f"Cosine Std = {cosine_std:.2f}\n"
    )


    ####################################################################################
    # Plot Histogram
    plt.figure(figsize=(8, 5))
    print(euclidean_distances)
    plt.xticks(fontsize=18)
    sns.kdeplot(euclidean_distances, color=align_color, fill=True, linewidth=1.5)
    plt.xlim([0, 1.5])   
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.savefig(f'histogram_{ALIGN_METHOD}.png',  bbox_inches='tight')  

    return

if __name__ == '__main__':
    if TRAIN == True:
        train(FRESH)
    else:
        evaluate()