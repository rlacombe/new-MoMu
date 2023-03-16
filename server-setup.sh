pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cu116.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.13.1+cu116.html

# set up data.
mkdir raw_data 
mkdir -p raw_data/XL

mkdir -p raw_model/text
mkdir -p raw_model/graph
python download_data.py

# unzip data
cd raw_data/contrastive_pretrain_data/xl/ 
unzip raw_data/contrastive_pretrain_data/xl/text-clipped.zip
unzip raw_data/contrastive_pretrain_data/xl/graph.zip
cd ../../../
