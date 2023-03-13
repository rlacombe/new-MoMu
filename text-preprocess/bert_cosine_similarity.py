import torch
from transformers import BertTokenizer, BertModel
import os
import pandas as pd
import glob

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

folder_path = 'data/contrast-pretrain/S/text/*.txt'
file_extension = '*.txt'

# Load synonyms cvs file as dataframe
pubchem_synonyms_csv = os.path.join('data', 'PubChem_synonyms_list.csv')

# Read pubchem synonyms CSV file into Pandas dataframe
pubchem_synonyms_df = pd.read_csv(pubchem_synonyms_csv)
pubchem_synonyms_df = pubchem_synonyms_df.sort_values(by='cid').reset_index(drop=True)

# Loop over txt files in folder
for file_path in glob.glob(folder_path):
    # Print file name
    txt_name = os.path.basename(file_path)
    cid = int(txt_name[len('text_'):-4])
    molecule_name = pubchem_synonyms_df.loc[pubchem_synonyms_df['cid'] == 8]['cmpdname'].iat[0]
    molecule_synonyms = pubchem_synonyms_df.loc[pubchem_synonyms_df['cid'] == 8]['cmpdsynonym'].iat[0].split('|')
    if molecule_name not in molecule_synonyms:
        print('Not there!')
        molecule_all = [molecule_name] + molecule_synonyms
    else:
        molecule_all = molecule_synonyms
    with open(file_path, 'r', encoding='utf-8') as f:
        # Read contents of file
        document = f.read()

    paragraphs = document.split('\n')[:-1]
    # Choose target word
    for molecule in molecule_all:
        target_word = molecule

        # Tokenize target word and document
        target_word_tokens = tokenizer.tokenize(target_word)
        for i in range(len(paragraphs)):
            paragraph_tokens = tokenizer.tokenize(paragraphs[i])
            # Check if document_tokens is empty
            if len(paragraph_tokens) == 0:
                print("Error: Empty input sequence.")
                continue

            # Convert tokens to tensor and add batch dimension
            target_word_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(target_word_tokens)]).long()
            paragraph_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(paragraph_tokens)]).long()

            # Get BERT embeddings for target word and document
            with torch.no_grad():
                target_word_embedding = model(target_word_tensor)[0][:, 0, :]
                paragraph_embedding = model(paragraph_tensor)[0][:, 0, :]

            similarity = torch.cosine_similarity(target_word_embedding, paragraph_embedding, dim=1)
            print(f"Paragraph {i + 1}, molecule name:{molecule} similarity score: {similarity.item()}")

print('Finished')
