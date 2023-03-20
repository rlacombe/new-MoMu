import gdown
import os

downloads = {
  'cos_sim_max-t0.05-eps0.5': 'https://drive.google.com/drive/u/1/folders/1-IGAw-8aWExDMkAIwSQ0QTWsPkaKUeOS',
  'cos_sim_max-t0.1-eps0.5': 'https://drive.google.com/drive/u/1/folders/1-QZLSMx8hFW2yJmGJxl-WkB8NLqhMOsK',
  'cos_sim_max-t0.2-eps0.5': 'https://drive.google.com/drive/u/1/folders/1-iq_d5jZAbmPRsvHi0vFAa-0hRhCmShB',
  'cos_sim_mean-t0.05-eps0.5': 'https://drive.google.com/drive/u/1/folders/1-DyDEoeJAGFvapXnD0e8tzqgOD_NoFqo',
  'cos_sim_mean-t0.1-eps0.5': 'https://drive.google.com/drive/u/1/folders/1-3UBcWbwKw7wCeDT0ZI10VtJ7cnNyQ50',
  'cos_sim_mean-t0.2-eps0.5': 'https://drive.google.com/drive/u/1/folders/1-Y6Fg0tVVHDP56JYxNOCE4QLalend4vC',
  'cos_sim_sent-t0.05-eps0.5': 'https://drive.google.com/drive/u/1/folders/1-gSCpjZsjFNC-Uh_P7qZQvoGitZydrDi',
  'cos_sim_sent-t0.1-eps0.5': 'https://drive.google.com/drive/u/1/folders/1-pNhBzuWOb8PNcffpKiMZzrlKxP-2xro',
  'cos_sim_sent-t0.2-eps0.5': 'https://drive.google.com/drive/u/1/folders/103cnba1-TBgPRmJmUNzRe2ZmYxNCsxYk' 

}

graph_augs = 'dnodes-subgraph-'
for folder_name, url in downloads.items():
    outpath = os.path.join('all_checkpoints', graph_augs+folder_name) 
    gdown.download_folder(url, output=outpath, quiet=False)
