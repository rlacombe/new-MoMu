DATASET_NAME=muv
DATASET_PATH=dataset/muv/muv.csv
graph_augs=dnodes-subgraph
sampling_methods=(cos_sim_mean cos_sim_max cos_sim_sent)
temps=(0.2 0.1 0.05)
eps=0.5
for sampling_method in "${sampling_methods[@]}"
do
  for temp in "${temps[@]}"
  do
    echo $temp
    echo $sampling_method
    model_path="all_checkpoints/$graph_augs-$sampling_method-$temp-$eps/best-ckpt.ckpt"
    echo $model_path
    python finetune.py \
	--num_workers 2 \
	--split "scaffold" \
	--gnn_type gin \
	--dataset $DATASET_NAME \
	--dataset_path $DATASET_PATH \
	--lr 1e-3 \
	--epochs 15 \
	--input_model_file $model_path 
  done
done
