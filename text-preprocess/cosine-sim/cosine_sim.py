import os
import torch
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Clip dataset to max 500 paragraphs per molecule.')
parser.add_argument('--query', '-q', type=str, help='The directory containing the molecule embeddings')
parser.add_argument('--key', '-k', type=str, help='The directory containing the paragraph embeddings')
args = parser.parse_args()

# Specify paths
query_dir= args.query
key_dir= args.key

query_files = os.listdir(query_dir)

# Loop through all the files in the source directory
for i, query_filename in enumerate(query_files):
    if query_filename.endswith('.pt'):
        
        # Get molecule CID
        cid = int(query_filename.split("_")[-1].split(".")[0])
        print(f"Processing text file for molecule {cid}: number {i+1} of {len(query_files)}.")

        # Load the query tensors
        query_file = os.path.join(query_dir, query_filename)
        query = torch.load(query_file)

        # Load the key tensor
        key_filename = f"embeds_{cid}.pt"
        key_file = os.path.join(key_dir, key_filename)
        key = torch.load(key_file)
        
        # Check dimension
        if not(query.shape[0] == key.shape[1]): break

        # Compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(query, key, dim=1)

        # Save it as CSV 
        scores = pd.DataFrame({'cosine_similarity': cos_sim.tolist()})
        scores.index.name = 'para_num'
        scores.reset_index(inplace=True)

        # save the DataFrame as a CSV file
        scores.to_csv(f"cos_sim_{cid}.csv", index=False)

        print(f"Done with molecule {i+1} of {len(query_files)}.\n")
 