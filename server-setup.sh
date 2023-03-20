pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cu116.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.13.1+cu116.html

# set up data
#mkdir -p data/contrast-pretrain/XL
#mkdir -p models/pretrained_gin
#mkdir -p models/pretrained_bert

# download data
#python download_data.py

# unzip data
#cd data/contrast-pretrain/XL
#mkdir graph
#mv graph.zip graph/
#cd graph/
#unzip graph.zip
#cd ../
#mkdir text
#mv text-clipped.zip text/
#cd text/
#unzip text-clipped.zip
#cd ../../../../
