import gdown
import os


downloads = {
  'contrastive_pretrain_xl': ('https://drive.google.com/drive/u/1/folders/1rNCHkpN_8LydAOEbCUXBqWAsBfYrBd12', 'folder', 'constrastive_pretrain_data/xl'),
  'gin_weights': ('https://drive.google.com/drive/u/1/folders/127pBYL6U9kym5h5NRyy-fO1ouKG4KS4y', 'folder', 'pretrained_models/graph_encoders/gin'),
  'bert_weights': ('https://drive.google.com/drive/u/1/folders/1SH871YWz1ViS5JwyIgaRCKJdvE7mFKex', 'folder', 'pretrained_models/text_encoders/bert')
}

for url, dtype, outpath in downloads.values():
    outpath = os.path.join('raw_data', outpath)
    if dtype == 'folder':
        gdown.download_folder(url, output=outpath, quiet=False)
    elif dtype == 'file':
        gdown.download(url, output=outpath, quiet=False)
    else:
        raise ValueError("invalid download type")
