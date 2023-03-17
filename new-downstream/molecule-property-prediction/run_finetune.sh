DATASET_NAME=muv
DATASET_PATH=muv.csv
python finetune.py \
	--input_model_file "./models/best-ckpt.ckpt" \
	--split "scaffold" \
	--gnn_type gin \
	--dataset $DATASET_NAME \
	--dataset_path $DATASET_PATH \
	--lr 1e-3 \
	--epochs 300
