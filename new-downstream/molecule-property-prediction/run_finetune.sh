graph_augs=dnodes-subgraph
sampling_methods=(cos_sim_mean cos_sim_max cos_sim_sent)
temps=(0.2 0.1 0.05)
eps=0.5
dataset_names=(muv hiv bace bbbp tox21 toxcast sider clintox)
epochs=(15 30 100 100 50 50 100 100) # these are assigned according to how big the datasets are.


for (( i=0; i<${#dataset_names[@]}; i++ ));
do
  dataset_name=${dataset_names[i]}
  num_epochs=${epochs[i]}
  dataset_path=dataset/$dataset_name/raw/$dataset_name.csv
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
  	--dataset $dataset_name \
  	--dataset_path $dataset_path \
  	--lr 1e-3 \
  	--epochs $num_epochs \
  	--input_model_file $model_path 
    done
  done
 done
