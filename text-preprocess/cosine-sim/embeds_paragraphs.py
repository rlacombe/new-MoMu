import os
import argparse
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

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

'''
# TODO: implement GPU and move data/model to CUDA device
'''

# Loop through each text file in the directory
for i, filename in enumerate(files):

    if filename.endswith(".txt"):

        embeds = pd.DataFrame(columns=["cid", "para_num", "para_embed"])    

        # Get molecule CID
        cid = int(filename.split("_")[1].split(".")[0])
        print(f"Processing text file for molecule {cid}: number {i+1} of {len(files)}.")

        # Loop through each paragraph in the file
        paragraphs = open(os.path.join(dir_path, filename))
        for l, paragraph in enumerate(paragraphs):

            # Tokenize paragraph
            paragraph_tokens = tokenizer.tokenize(paragraph)
            paragraph_tokens = paragraph_tokens[:max_bert_token_length]

            # Check if document_tokens is empty
            if len(paragraph_tokens) == 0:
                print("Error: Empty input sequence.")
                continue

            # Convert tokens to tensor and add batch dimension
            paragraph_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(paragraph_tokens)]).long()
        
            # Get BERT embeddings for paragraph
            with torch.no_grad():
                paragraph_embedding = model(paragraph_tensor)[0][:, 0, :]
                embeds = pd.concat([embeds, pd.DataFrame({'cid': cid, "para_num": l, "para_embed": [paragraph_embedding]}, index=[0])], ignore_index=True)                                    

            print(f"Done with paragraph {l}.")

        print(f"Done with molecule {i} of {len(files)}.\n")
                    
        # Save the results to a CSV file
        embeds.to_csv(f"embeds_{cid}.csv", index=False)

print('Finished')


