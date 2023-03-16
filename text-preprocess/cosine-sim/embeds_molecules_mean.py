import os
import argparse
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence

parser = argparse.ArgumentParser(description='Extract embeddings of molecules listed in a directory.')
parser.add_argument('--directory', '-d', type=str, default='../../data/contrast-pretrain/S/text', help='The directory containing the text files')
parser.add_argument('--names', '-n', type=str, default='../../data/cosine-sim/PubChem_synonyms_list.csv', help='The file containing the molecule names')
parser.add_argument('--top_k', '-k', type=int, default=10, help='How many synonyms to use')
args = parser.parse_args()

# Specify path, initialize files and store results
dir_path= args.directory
files = os.listdir(dir_path)
num_files = len(files)

names_path = args.names
top_k = args.top_k

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
max_bert_token_length = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model.to(device)

# Get molecule synonyms
with open(names_path, 'r') as f:
    pubchem_synonyms_df = pd.read_csv(names_path)
    #pubchem_synonyms_df = pubchem_synonyms_df.sort_values(by='cid').reset_index(drop=True)

# Loop through each text file in the directory
for i, filename in enumerate(files):

    if filename.endswith(".txt"):

        # Get molecule CID
        cid = int(filename.split("_")[1].split(".")[0])
        print(f"Processing molecule {cid}: number {i+1} of {len(files)}.")

        # Loop through each synonym for molecule cid
        molecule_name = pubchem_synonyms_df.loc[pubchem_synonyms_df['cid'] == cid]['cmpdname'].iat[0]
        molecule_synonyms = pubchem_synonyms_df.loc[pubchem_synonyms_df['cid'] == cid]['cmpdsynonym'].iat[0].split('|')
    
        #if molecule_name not in molecule_synonyms:
        synonyms = [molecule_name] + molecule_synonyms
        #else:
        #    synonyms = molecule_synonyms

        # Keep only the name + top k synonyms
        synonyms = synonyms[:1+top_k]   

        name_tensors_list = []

        for n, name in enumerate(synonyms): 

            # Tokenize paragraph
            name_tokens = tokenizer.tokenize(name)
            name_tokens = name_tokens[:max_bert_token_length]

            # Check if name_tokens is empty
            if len(name_tokens) == 0:
                print("Error: empty synonyms list.")
                continue

            # Convert tokens to tensor  
            name_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(name_tokens)]).int()
            
            name_tensors_list.append(name_tensor.T)                
            print(f"Done with synonym {n}.")

        # Pad and stack tensors along the batch dimension
        names_tensor = pad_sequence(name_tensors_list, batch_first=True).squeeze(2).to(device)
        print(names_tensor.shape)

        # Get BERT embeddings for all names tensor
        with torch.no_grad():
            names_embeddings = model(names_tensor)[0][:, 0, :].to(device)

        # Compute mean along first dimension
        query_vector = 0.5*names_embeddings[0]+0.5*torch.mean(names_embeddings[1:11], dim=0)

        # Save the results to a PT file
        torch.save(query_vector, f"query_mean_{cid}.pt")
        print(f"Done with molecule {i+1} of {len(files)}.\n")
            
print('Finished')