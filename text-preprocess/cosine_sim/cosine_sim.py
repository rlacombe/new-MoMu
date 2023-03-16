import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence


def paragraph_embeds(text_file, tokenizer, model, device, max_para_length, max_bert_token_length):

    """
    Given a text file containing paragraphs, this method tokenizes each paragraph, converts the tokens to a PyTorch tensor,
    and then applies a pre-trained model to get the BERT embeddings for each paragraph. It returns a tensor of the embeddings
    for all the paragraphs.

    :param text_file: A string representing the path to the input text file.
    :param tokenizer: A pre-trained tokenizer object that will be used to tokenize the paragraphs.
    :param model: A pre-trained BERT model that will be used to generate the paragraph embeddings.
    :param device: The device on which the computation will be performed. Must be a string: either "cpu" or "cuda".
    :param max_para_length: An integer representing the maximum length (in characters) of each paragraph.
    :param max_bert_token_length: An integer representing the maximum number of BERT tokens that will be used to represent each paragraph.

    :return: A PyTorch tensor of shape (n, d), where n is the number of paragraphs in the input file and d is the dimensionality
    of the BERT embeddings.
    """

    # Loop through each paragraph in the file        
    with open(text_file, 'r') as source_file:

        # Read the first 500 lines of the file
        paragraphs = source_file.readlines()[:500]
        paragraph_tensors_list = []

        for l, paragraph in enumerate(paragraphs): 

            # Tokenize paragraph
            paragraph_tokens = tokenizer.tokenize(paragraph[:max_para_length])  # 'TODO' Model loads only first 256 words
            paragraph_tokens = paragraph_tokens[:max_bert_token_length]

            # Check if document_tokens is empty
            if len(paragraph_tokens) == 0:
                print("Error: Empty input sequence.")
                continue

            # Convert tokens to tensor  
            paragraph_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(paragraph_tokens)]).int()
            paragraph_tensors_list.append(paragraph_tensor.T)                
            print(f"Done with paragraph {l}.")

        # Pad and stack tensors along the batch dimension
        paragraph_tensors = pad_sequence(paragraph_tensors_list, batch_first=True).squeeze(2).to(device)
            
        # Get BERT embeddings for full tensor
        with torch.no_grad():
            paragraph_embeddings = model(paragraph_tensors)[0][:, 0, :].to(device)

        # Save the results to a PT file
    return paragraph_embeddings


def get_molecule_synonyms(cid, pubchem_synonyms_df, top_k=10):

    """
    Given a compound identifier (cid) and a pandas DataFrame of PubChem synonyms for compounds (pubchem_synonyms_df),
    returns a list of compound synonyms including the compound name and the top k synonyms (default k=10).

    Args:
        cid (int): The compound identifier.
        pubchem_synonyms_df (pd.DataFrame): The pandas DataFrame containing PubChem synonyms for compounds.
        top_k (int, optional): The number of top synonyms to include in the output list. Defaults to 10.

    Returns:
        List[str]: A list of compound synonyms, including the compound name and the top k synonyms.
    """

    molecule_name = pubchem_synonyms_df.loc[pubchem_synonyms_df['cid'] == cid]['cmpdname'].iat[0]
    molecule_synonyms = pubchem_synonyms_df.loc[pubchem_synonyms_df['cid'] == cid]['cmpdsynonym'].iat[0].split('|')

    if molecule_name not in molecule_synonyms:
        synonyms = [molecule_name] + molecule_synonyms
    else:
        synonyms = molecule_synonyms

    # Keep only the name + top k synonyms
    synonyms = synonyms[:1+top_k]   

    return synonyms


def get_molecule_embeds_mean(synonyms, tokenizer, model, device, max_bert_token_length, syn_weight=0.7):

    """
    Given a list of molecule synonyms, a tokenizer, a pre-trained language model, a device to run the model on, and a maximum length for the BERT tokens, returns the mean embedding of the molecule synonyms.

    Args:
        synonyms (List[str]): A list of molecule synonyms.
        tokenizer (transformers.PreTrainedTokenizer): A tokenizer to tokenize the synonyms.
        model (transformers.PreTrainedModel): A pre-trained language model to obtain embeddings.
        device (torch.device): The device to run the model on.
        max_bert_token_length (int): The maximum number of BERT tokens allowed.
        syn_weight (float, optional): The weight to give to the remaining synonyms when computing the mean embedding. Defaults to 0.7.

    Returns:
        torch.Tensor: The mean embedding of the molecule synonyms.
    """

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
    molecule_embeds = (1-syn_weight)*names_embeddings[0]+syn_weight*torch.mean(names_embeddings[1:11], dim=0)

    return molecule_embeds