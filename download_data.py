import gdown
import os


downloads = {
  'contrastive_pretrain_xl': ('https://drive.google.com/drive/u/1/folders/1rNCHkpN_8LydAOEbCUXBqWAsBfYrBd12', 'folder', 'data/contrast-pretrain/XL'),
  'gin_weights': ('https://drive.google.com/drive/u/1/folders/127pBYL6U9kym5h5NRyy-fO1ouKG4KS4y', 'folder', 'models/pretrained_gin'),
  'bert_weights': ('https://drive.google.com/drive/u/1/folders/1SH871YWz1ViS5JwyIgaRCKJdvE7mFKex', 'folder', 'models/pretrained_bert')
}

for url, dtype, outpath in downloads.values():
    if dtype == 'folder':
        gdown.download_folder(url, output=outpath, quiet=False)
    elif dtype == 'file':
        gdown.download(url, output=outpath, quiet=False)
    else:
        raise ValueError("invalid download type")
