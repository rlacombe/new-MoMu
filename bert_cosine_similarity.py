import torch
from transformers import BertTokenizer, BertModel
import os
import pandas as pd
import glob
import shutil
import numpy as np

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
max_bert_token_lenght = 512

folder_to_save = 'data/contrast-pretrain/S/updated_text/'
old_file_path = 'data/contrast-pretrain/S/text/'
folder_path = 'data/contrast-pretrain/S/text/*.txt'  # Change folder path to run on XL data if needed
file_extension = '*.txt'
synonyms = False
# Load synonyms cvs file as dataframe
pubchem_synonyms_csv = os.path.join('data', 'PubChem_synonyms_list.csv')

# Read pubchem synonyms CSV file into Pandas dataframe
pubchem_synonyms_df = pd.read_csv(pubchem_synonyms_csv)
pubchem_synonyms_df = pubchem_synonyms_df.sort_values(by='cid').reset_index(drop=True)
# Loop over txt files in folder
for file_path in glob.glob(folder_path):

    # Create an empty dataframe
    df = pd.DataFrame(columns=['paragraph', 'number', 'molecule', 'score'])
    paragraphs_df = pd.DataFrame(columns=['paragraph','number'])
    molecules_df = pd.DataFrame(columns=['molecule_name', 'molecule_embedding'])
    # Print file name
    txt_name = os.path.basename(file_path)
    if os.path.exists(os.path.join(folder_to_save, txt_name)):
        print(f"{txt_name} exists in {folder_to_save}")
        continue

    cid = int(txt_name[len('text_'):-4])
    molecule_name = pubchem_synonyms_df.loc[pubchem_synonyms_df['cid'] == cid]['cmpdname'].iat[0]
    molecule_synonyms = pubchem_synonyms_df.loc[pubchem_synonyms_df['cid'] == cid]['cmpdsynonym'].iat[0].split('|')
    if molecule_name not in molecule_synonyms:
        # print('Not there!')
        molecule_all = [molecule_name] + molecule_synonyms
    else:
        molecule_all = molecule_synonyms
    with open(file_path, 'r', encoding='utf-8') as f:
        # Read contents of file
        document = f.read()

    paragraphs = document.split('\n')[:-1]
    if len(paragraphs) < 6:  # If only 6 paragraphs or less, keep all paragraphs
        old_path = os.path.join(old_file_path, txt_name)
        new_path = os.path.join(folder_to_save, txt_name)
        shutil.copyfile(old_path, new_path)
        continue
    # Choose target word


    for i in range(len(paragraphs)):
        paragraph_tokens = tokenizer.tokenize(paragraphs[i])
        # Check if document_tokens is empty
        if len(paragraph_tokens) == 0:
            print("Error: Empty input sequence.")
            continue

        # Convert tokens to tensor and add batch dimension
        paragraph_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(paragraph_tokens)]).long()
        if paragraph_tensor.size(1) > max_bert_token_lenght:
            print("Error: Paragraph too long")  # Might fix later - split the long paragraph, tokenize then merge
            continue
        # Get BERT embeddings for target word and document
        with torch.no_grad():
            paragraph_embedding = model(paragraph_tensor)[0][:, 0, :]
            paragraphs_df_new = pd.DataFrame({'paragraph': [paragraph_embedding], 'number' : [i]})
            paragraphs_df = pd.concat([paragraphs_df, paragraphs_df_new], ignore_index=True)
        print(f"Done with paragraph {i} of {len(paragraphs)}")
        if i > 100:
            break

    for j in range(len(molecule_all)):
        molecule = molecule_all[j]
        # if counter > 5:
        #     continue
        target_word = molecule

        # Tokenize target word and document
        target_word_tokens = tokenizer.tokenize(target_word)
        # Convert tokens to tensor and add batch dimension
        target_word_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(target_word_tokens)]).long()

        # Get BERT embeddings for target word and document
        with torch.no_grad():
            target_word_embedding = model(target_word_tensor)[0][:, 0, :]
            molecules_df_new = pd.DataFrame({'molecule_name': [molecule], 'molecule_embedding': [target_word_embedding]})
            molecules_df = pd.concat([molecules_df, molecules_df_new], ignore_index=True)
        print(f"Done with molecule {j} of {len(molecule_all)}")
        if j > 10:
            break
    # Everything above works. Everything below is a work in progress.
    # Convert the "paragraph" column to a list of tensors
    paragraph_tensors = [tensor for tensor in paragraphs_df['paragraph']]

    # Stack the tensors along a new dimension to create a 2D tensor
    paragraph_embeddings = torch.stack(paragraph_tensors, dim=0)

    # Convert the "molecule_embedding" column to a list of tensors
    molecule_tensors = [tensor for tensor in molecules_df['molecule_embedding']]

    # Stack the tensors along a new dimension to create a 2D tensor
    molecule_embeddings = torch.stack(molecule_tensors, dim=0)

    # Cosine similarity
    similarity_matrix = torch.cosine_similarity(paragraph_embeddings.unsqueeze(1), molecule_embeddings.unsqueeze(0),
                                                dim=2)
    average_similarity = torch.mean(similarity_matrix, dim=1)
    paragraphs_df['average_similarity'] = average_similarity.detach().numpy()

    print('Done with {}'.format(txt_name))

print('Finished')
