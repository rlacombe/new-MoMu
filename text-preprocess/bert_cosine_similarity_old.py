import torch
from transformers import BertTokenizer, BertModel
import os
import pandas as pd
import glob
import shutil

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
max_bert_token_lenght = 512

folder_to_save = 'data/contrast-pretrain/S/updated_text/'
old_file_path = 'data/contrast-pretrain/S/text/'
folder_path = 'data/contrast-pretrain/S/text/*.txt'  # Change folder path to run on XL data if needed
file_extension = '*.txt'

# Load synonyms cvs file as dataframe
pubchem_synonyms_csv = os.path.join('data', 'PubChem_synonyms_list.csv')

# Read pubchem synonyms CSV file into Pandas dataframe
pubchem_synonyms_df = pd.read_csv(pubchem_synonyms_csv)
pubchem_synonyms_df = pubchem_synonyms_df.sort_values(by='cid').reset_index(drop=True)
# Loop over txt files in folder
# for file_path in glob.glob(folder_path):
for file_path in glob.glob(folder_path):

    # Create an empty dataframe
    df = pd.DataFrame(columns=['paragraph', 'number', 'molecule', 'score'])
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
    counter = 0
    for molecule in molecule_all:
        if counter > 5:
            continue
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
            if paragraph_tensor.size(1) > max_bert_token_lenght:
                print("Error: Paragraph too long")  # Might fix later - split the long paragraph, tokenize then merge
                continue
            # Get BERT embeddings for target word and document
            with torch.no_grad():
                target_word_embedding = model(target_word_tensor)[0][:, 0, :]
                paragraph_embedding = model(paragraph_tensor)[0][:, 0, :]

            similarity = torch.cosine_similarity(target_word_embedding, paragraph_embedding, dim=1)
            df_new = pd.DataFrame({'paragraph': [paragraphs[i]],
                                   'number': [i + 1],
                                   'molecule': [molecule],
                                   'score': [similarity.item()]})
            df = pd.concat([df, df_new], ignore_index=True)
            print(
                f"Paragraph {i + 1}, molecule name:{molecule} similarity score: {similarity.item()}, text file: {txt_name}")
            if i > 100:
                break
        counter = 1 + counter


    df = df.sort_values('score', ascending=False)
    df = df.drop_duplicates(subset='paragraph').head(5)
    save_txt = os.path.join(folder_to_save, txt_name)
    with open(save_txt, 'w', encoding='utf-8') as file:
        file.write('\n'.join(df['paragraph'].str.strip().tolist()))
    print('Done with {}'.format(txt_name))

print('Finished')
