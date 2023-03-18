import os
import nltk
import argparse
import pandas as pd
nltk.download('punkt')
import cosine_sim.cosine_sim as cs

parser = argparse.ArgumentParser(description='Keep only relevant sentences.')
parser.add_argument('--names', '-n', type=str, help='The path of the file containing the molecule names')
parser.add_argument('--source_dir', '-s', type=str, help='The source directory containing the text files')
parser.add_argument('--target_dir', '-t', type=str, help='The target directory where to save the new text files')
parser.add_argument('--top_k', '-k', type=int, default=20,  help='Number of top-k synonyms to compute cosine similarity (default: 20)')

# Specify paths
args = parser.parse_args()

names_file = args.names
source_dir= args.source_dir
target_dir= args.target_dir
top_k = args.top_k

files = os.listdir(source_dir)

pubchem_synonyms_df = pd.read_csv(names_file)

# Loop through all the files in the source directory
for i, filename in enumerate(files):
    if filename.endswith('.txt'):

        # Open the source file
        with open(os.path.join(source_dir, filename), 'r') as source_file:

            # Read the first 500 lines of the file
            first_500_lines = source_file.readlines()[:500]

        # Get the molecule name and synonyms
        cid = int(filename.split("_")[-1].split(".")[0])
        synonyms = cs.get_molecule_synonyms(cid, pubchem_synonyms_df, top_k=top_k)

        # Initialize working list of relevant paragraphs in text file
        relevant_paragraphs_list = []

        # Loop over each paragraph in text file
        for line in first_500_lines:  

            # Cut paragraph into sentences
            sentences = nltk.sent_tokenize(line)
            
            # Initialize list of relevant
            relevant_sentences_list = []

            # Loop over each sentence in the paragraph  
            for sentence in sentences:

                count = 0
                sentence_lower = sentence.lower()  # Convert everything to lowercase
                
                # Loop over synonyms
                for name in synonyms:
                    count += sentence_lower.count(name.lower())
                    if count > 0: # Found the name or synonym
                        relevant_sentences_list.append(sentence) 
                        break

            # If working paragraph has a relevant sentence: append to working text file
            if not relevant_sentences_list == []:
                relevant_paragraphs_list.append(" ".join(relevant_sentences_list))

        # If paragraph list is empty, just copy the file
        if relevant_paragraphs_list == []:
            with open(os.path.join(target_dir, filename), 'w') as target_file:
                for paragraph in first_500_lines:
                    target_file.write(paragraph + '\n')
        else:  # Write to the target file
            with open(os.path.join(target_dir, filename), 'w') as target_file:
                for paragraph in relevant_paragraphs_list:
                    target_file.write(paragraph + '\n')

        print(f"File {i+1} out of {len(files)}.")