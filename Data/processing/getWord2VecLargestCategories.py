import gensim.downloader as api
import csv
import numpy as np
import torch

# Load Word2Vec model
wv = api.load("word2vec-google-news-300")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read categories and weights
categories_reader = csv.reader(open("/home/gmc62/NSD/largest_categories.csv", "r"))
weights_reader = csv.reader(open("/home/gmc62/NSD/largest_categories_weights.csv", "r"))

def Word2VecEmbed(categories, weights):
    embeds = []
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    for category in categories:
        if len(category.split()) == 2:
            cat1, cat2 = category.split()
            embed1 = torch.tensor(wv[cat1]).to(device)
            embed2 = torch.tensor(wv[cat2]).to(device)
            embed = torch.mean(torch.stack([embed1, embed2]), dim=0)
            embeds.append(embed)
        else:
            embeds.append(torch.tensor(wv[category]).to(device))

    embeds_tensor = torch.stack(embeds).to(device)
    weighted_embeds = weights_tensor.unsqueeze(1) * embeds_tensor  # Apply weights
    sentence_embedding = torch.sum(weighted_embeds, dim=0)  # Weighted sum
    sentence_embedding = sentence_embedding.reshape(1, 300).to(device)
    return sentence_embedding

if __name__ == "__main__":
    embed = torch.empty((0, 300)).to(device)
    
    for i, (categories, weights) in enumerate(zip(categories_reader, weights_reader)):
        print(i)
        # Strip and clean category names
        categories = [category.strip() for category in categories if category.strip()]
        
        # Parse weights as floats
        weights = [float(weight) for weight in weights if weight.strip()]
        
        sentence_embedding = Word2VecEmbed(categories, weights)
        embed = torch.cat((embed, sentence_embedding), dim=0).to(device)
    
    torch.save(embed, "/home/gmc62/NSD/largest_categories_embed.pth")
