# import gensim.downloader as api
# import csv
# import numpy as np
# import torch
# from sklearn.metrics.pairwise import cosine_similarity

# wv = api.load("word2vec-google-news-300")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def Word2VecEmbed(text):
#     tokens = text.split()
#     print(tokens)
#     word_vectors = [torch.tensor(wv[word]).to(device) for word in tokens if word in wv]
#     print(word_vectors)
#     if word_vectors:
#         sentence_embedding = torch.mean(torch.stack(word_vectors), dim=0).to(device)
#         sentence_embedding = sentence_embedding.reshape(1, 300).to(device)
#         return sentence_embedding
#     else:
#         return torch.zeros(1, 300).to(device)

# if __name__ == "__main__":
#     text1 = 'apple'
#     text2 = 'banana'
#     embed1 = Word2VecEmbed(text1).detach().cpu().numpy()
#     embed2 = Word2VecEmbed(text2).detach().cpu().numpy()
#     similarity_score = cosine_similarity(embed1, embed2)
#     print(f"Cosine Similarity: {similarity_score}")
from gensim.models import KeyedVectors
import gensim.downloader as api

# Load the Word2Vec model using Gensim's API
print("Downloading and loading the Word2Vec model...")
wv = api.load("word2vec-google-news-300")

def get_most_similar_words(word, topn=10):
    try:
        # Check if the word is in the model's vocabulary
        if word in wv:
            similar_words = wv.most_similar(word, topn=topn)
            return similar_words
        else:
            return f"The word '{word}' is not in the model vocabulary."
    except Exception as e:
        return f"An error occurred: {e}"

# Example: Find the most similar words to 'king'
word = 'microwave'
topn = 20  # Number of similar words to return

similar_words = get_most_similar_words(word, topn)
if isinstance(similar_words, list):
    print(f"Most similar words to '{word}':")
    for similar_word, similarity in similar_words:
        print(f"  {similar_word}: {similarity}")
else:
    print(similar_words)

