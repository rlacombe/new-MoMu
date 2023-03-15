import os
import argparse
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence

parser = argparse.ArgumentParser(description='Extract embeddings of paragraphs in file texts.')
parser.add_argument('--directory', '-d', type=str, default='../../data/S/text', help='The directory containing the text files')
args = parser.parse_args()

# Specify path, initialize files and store results
dir_path= args.directory
files = os.listdir(dir_path)
num_files = len(files)

embeds = pd.DataFrame(columns=["cid", "para_num", "para_embed"])

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
max_bert_token_length = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

tokenizer
model.to(device)

# Loop through each text file in the directory
for i, filename in enumerate(files):

    if filename.endswith(".txt"):

        embeds = pd.DataFrame(columns=["cid", "para_num", "para_embed"])   

        # Get molecule CID
        cid = int(filename.split("_")[1].split(".")[0])
        print(f"Processing text file for molecule {cid}: number {i+1} of {len(files)}.")

        # Loop through each paragraph in the file        
        with open(os.path.join(dir_path, filename), 'r') as source_file:

            # Read the first 500 lines of the file
            paragraphs = source_file.readlines()[:500]
            paragraph_tensors_list = []

            for l, paragraph in enumerate(paragraphs): 

                # Tokenize paragraph
                paragraph_tokens = tokenizer.tokenize(paragraph)
                paragraph_tokens = paragraph_tokens[:max_bert_token_length]

                # Check if document_tokens is empty
                if len(paragraph_tokens) == 0:
                    print("Error: Empty input sequence.")
                    continue

                # Convert tokens to tensor  
                paragraph_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(paragraph_tokens)]).long()
                #print(paragraph_tensor.shape)
                paragraph_tensors_list.append(paragraph_tensor.T)                
                print(f"Done with paragraph {l}.")

            # Pad and stack tensors along the batch dimension
            paragraph_tensors = pad_sequence(paragraph_tensors_list, batch_first=True).squeeze(2).to(device)
            print(paragraph_tensors.shape)
            #paragraph_tensors = torch.stack(padded_tensors, dim=0).to(device)

            # Get BERT embeddings for full tensor
            with torch.no_grad():
                paragraph_embeddings = model(paragraph_tensors)[0][:, 0, :].to(device)

            # Save the results to a PT file
            torch.save(paragraph_embeddings, f"embeds_{cid}.pt")
            print(f"Done with molecule {i+1} of {len(files)}.\n")
            
print('Finished')