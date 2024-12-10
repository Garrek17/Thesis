import gensim.downloader as api
import csv
import numpy as np
import torch

wv = api.load("word2vec-google-news-300")
csv_reader = csv.reader(open("/home/gmc62/NSD/captions.csv", "r"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embed = torch.empty((0, 300)).to(device)

def Word2VecEmbed(text):
    tokens = text.split()
    word_vectors = [torch.tensor(wv[word]).to(device) for word in tokens if word in wv]
    if word_vectors:
        sentence_embedding = torch.mean(torch.stack(word_vectors), dim=0).to(device)
        sentence_embedding = sentence_embedding.reshape(1, 300).to(device)
        return sentence_embedding
    else:
        return torch.zeros(1, 300).to(device)

if __name__ == "__main__":
    for i, text in enumerate(csv_reader):
        print(i)
        marked_text = str(text[0])
        sentence_embedding = Word2VecEmbed(marked_text)
        embed = torch.cat((embed, sentence_embedding), dim=0).to(device)
    torch.save(embed, "/home/gmc62/NSD/captions_Word2Vec.pth")
