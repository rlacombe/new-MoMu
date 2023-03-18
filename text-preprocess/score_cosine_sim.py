import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
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
parser.add_argument('--method', '-m', type=str, default="all", help='Method with which to compute the scores (mean|max|sent|all)')
parser.add_argument('--top_k', '-k', type=int, default=20,  help='Number of top-k synonyms to compute cosine similarity (default: 20)')
parser.add_argument('--weight', '-w', type=int, default=0.8,  help='Weight given to synonyms vs main name (default: 0.8)')
parser.add_argument('--len', '-l', type=int, default=1024,  help='Context length (default: 1024)')


args = parser.parse_args()

# Specify arguments
text_dir= args.source_dir
scores_dir= args.target_dir
names_file = args.names
filetype = args.filetype
method = args.method
top_k = args.top_k
syn_weight = args.weight
context_len = args.len

# Load pre-trained BERT model and tokenizer on device
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

max_bert_token_length = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

model.to(device)

# Load molecule names file as a dataframe
pubchem_synonyms_df = pd.read_csv(names_file)

# Loop through all the files in the source directory
text_files = os.listdir(text_dir)

progress_bar = tqdm(enumerate(text_files), total=len(text_files), desc=f"In progress")

for i, text_filename in progress_bar:

    if text_filename.endswith('.txt'):
    
        # Get molecule CID
        cid = int(text_filename.split("_")[-1].split(".")[0])

        if os.path.isfile(os.path.join(scores_dir, f"scores_{cid}.pt")): continue

        progress_bar.set_description(f"Molecule {cid} ({i+1}/{len(text_files)})")

        # Retrieve paragraph embeddings
        paragraph_embeddings = cs.get_paragraph_embeds(os.path.join(text_dir, text_filename), \
                                        tokenizer, model, device, max_bert_token_length, context_len)
        
        # Load the molecule names
        synonyms = cs.get_molecule_synonyms(cid, pubchem_synonyms_df, top_k=top_k)

        # Retrieve the molecule name embeddings
        molecule_embeds = cs.get_molecule_embeddings(synonyms, tokenizer, model, device, max_bert_token_length)
        sentence_query_embeds = cs.get_sentence_query_embeddings(synonyms, tokenizer, model, device, max_bert_token_length)

        # Prepare to run through all methods at once
        cosine_mean = torch.tensor([])
        cosine_max = torch.tensor([])
        cosine_sent = torch.tensor([])

        # Compute cosine similarity for 'mean' method    
        if (method == "mean" or method == "all"):
            query = (1-syn_weight) * molecule_embeds[0] + syn_weight*torch.mean(molecule_embeds, dim=0)
            key = paragraph_embeddings
            if query.shape[0] == key.shape[1]:
                cosine_mean = torch.nn.functional.cosine_similarity(query, key, dim=1)

        # Compute cosine similarity for 'max' method    
        if (method == "max" or "all"):
            query = molecule_embeds
            key = paragraph_embeddings
            if query.shape[1] == key.shape[1]:
                cosine_max = torch.max(torch.nn.functional.cosine_similarity(query.unsqueeze(1), key.unsqueeze(0), dim=-1), dim=0)[0]

        # Compute cosine similarity for 'max' method    
        if (method == "sent" or method == "all"):
            query = sentence_query_embeds 
            key = paragraph_embeddings
            if query.shape[1] == key.shape[1]:
                cosine_sent = torch.nn.functional.cosine_similarity(query, key, dim=1) 

        # Save score according to the required format
        if (filetype == "csv" or filetype == "both"):
            scores = pd.DataFrame({'cosine_mean': cosine_mean.tolist(), 'cosine_max': cosine_max.tolist(),'cosine_sent': cosine_sent.tolist()})
            scores.index.name = 'para_num'
            scores.reset_index(inplace=True)
            scores.to_csv(os.path.join(scores_dir, f"scores_{cid}.csv"), index=False)

        if (filetype == "pt" or filetype == "both"):
            stacked = torch.stack([cosine_mean, cosine_max, cosine_sent], dim=0)
            torch.save(stacked, os.path.join(scores_dir, f"scores_{cid}.pt"))
