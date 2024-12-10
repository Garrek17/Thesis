from transformers import BertTokenizer, BertModel
import torch
import csv
import numpy as np
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
csv_reader = csv.reader(open("/home/gmc62/NSD/captions.csv", "r"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BertModel.from_pretrained('bert-base-uncased',
                                   output_hidden_states = True,
                                   )
model.to(device)
model.eval()
embed = torch.empty((0,768)).to(device)

def BERTEmbed(marked_text):
       tokenized_text = tokenizer.tokenize(marked_text)
       indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
       segments_ids = [1] * len(tokenized_text)
       tokens_tensor = torch.tensor([indexed_tokens]).to(device)
       segments_tensors = torch.tensor([segments_ids]).to(device)

       with torch.no_grad():
              outputs = model(tokens_tensor, segments_tensors)
              hidden_states = outputs[2]
              token_embeddings = torch.stack(hidden_states, dim=0).to(device)
              token_embeddings = torch.squeeze(token_embeddings, dim=1).to(
                     device)
              token_embeddings = token_embeddings.permute(1, 0, 2).to(device)
              token_vecs = hidden_states[-2][0]
              sentence_embedding = torch.mean(token_vecs, dim=0).to(device)
              sentence_embedding = torch.reshape(sentence_embedding,
                                                 (1, 768)).to(device)

              return sentence_embedding

if __name__ == "__main__":

       
       for i, text in enumerate(csv_reader):
              print(i)
              marked_text = "[CLS] " + str(text[0]) + " [SEP]"
              sentence_embedding = BERTEmbed(marked_text)
              embed = torch.cat((embed, sentence_embedding), dim=0).to(device)
       torch.save(embed, "/home/gmc62/NSD/captions_BERT.pth")

