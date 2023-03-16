import os
import torch
import argparse
import pandas as pd
import cosine_sim.cosine_sim as cs
from transformers import BertTokenizer, BertModel

"""
Script that takes inputs and computes the scores for a text corpus

Args:
    names (str): The path of the file containing the molecule names
    source_dir (str): The source directory containing the text files
    target_dir (str): The target directory where to save the score files
    format (str): Formats under which to save the scores (csv|pt|both)
    method (str): Method with which to compute the scores (mean|max|sent)

Returns:
    None: The script saves the computed cosine similarity scores as .pt and/or CSV files.
"""

parser = argparse.ArgumentParser(description='Compute scores.')
parser.add_argument('--names', '-n', type=str, help='The path of the file containing the molecule names')
parser.add_argument('--source_dir', '-s', type=str, help='The source directory containing the text files')
parser.add_argument('--target_dir', '-t', type=str, help='The target directory where to save the score files')
parser.add_argument('--filetype', '-f', type=str, default="both", help='Formats under which to save the scores (csv|pt|both)')
parser.add_argument('--method', '-m', type=str, default="mean", help='Method with which to compute the scores (mean|max|sent)')
parser.add_argument('--top_k', '-k', type=int, default=10,  help='Number of top-k synonyms to compute cosine similarity')

args = parser.parse_args()

# Specify arguments
text_dir= args.source_dir
scores_dir= args.target_dir
names_file = args.names
filetype = args.filetype
method = args.method
top_k = args.top_k

text_files = os.listdir(text_dir)

# Load pre-trained BERT model and tokenizer on device
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

max_para_length = 256 
max_bert_token_length = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model.to(device)

# Load molecule names file as a dataframe
pubchem_synonyms_df = pd.read_csv(names_file)

# Loop through all the files in the source directory
for i, text_filename in enumerate(text_files):
    if text_filename.endswith('.txt'):
        
        # Get molecule CID
        cid = int(text_filename.split("_")[-1].split(".")[0])
        print(f"Score for molecule {cid}: {i+1} of {len(text_files)}.")

        # Retrieve paragraph embeddings
        paragraph_embeddings = cs.paragraph_embeds(os.path.join(text_dir, text_filename), \
                                        tokenizer, model, device, max_para_length, max_bert_token_length)
        
        # Load the molecule names
        synonyms = cs.get_molecule_synonyms(cid, pubchem_synonyms_df, top_k=top_k)

        # Retrieve query embeddings and compute cosine similarity according the required method    
        if method == "mean":
            query = cs.get_molecule_embeds_mean(synonyms, tokenizer, model, device, max_bert_token_length, syn_weight=0.7)
            if not(query.shape[0] == paragraph_embeddings.shape[1]): break
            else:
                cos_sim = torch.nn.functional.cosine_similarity(query, paragraph_embeddings, dim=1)

        elif method == "max":
            pass

        elif method == "sent":
            pass

        # Save score according to the required format
        if (filetype == "csv" or filetype == "both"):
            scores = pd.DataFrame({'cosine_similarity': cos_sim.tolist()})
            scores.index.name = 'para_num'
            scores.reset_index(inplace=True)
            scores.to_csv(os.path.join(scores_dir, f"scores_{cid}.csv"), index=False)

        if (filetype == "pt" or filetype == "both"):
            torch.save(cos_sim, os.path.join(scores_dir, f"scores_{cid}.pt"))

        print(f"Done with molecule {i+1} of {len(text_files)}.\n") 