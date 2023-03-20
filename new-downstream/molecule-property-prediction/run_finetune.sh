graph_augs=dnodes-subgraph
sampling_methods=(cos_sim_mean cos_sim_max cos_sim_sent)
temps=(0.2 0.1 0.05)
eps=0.5
dataset_names=(bace bbbp tox21 toxcast sider clintox hiv muv)
epochs=(50 50 50 50 50 50 50 30 15) # these are assigned according to how big the datasets are.
seeds=(0 1 2 3 4 5 6 7 8 9)

for seed in "${seeds[@]}"
do
  for (( i=0; i<${#dataset_names[@]}; i++ ));
  do
    dataset_name=${dataset_names[i]}
    num_epochs=${epochs[i]}
    dataset_path=dataset/$dataset_name/raw/$dataset_name.csv
    for sampling_method in "${sampling_methods[@]}"
    do
      for temp in "${temps[@]}"
      do
	echo $seed
	echo $dataset_name
        echo $temp
        echo $sampling_method
        model_path="all_checkpoints/$graph_augs-$sampling_method-t$temp-eps$eps/best-ckpt.ckpt"
        echo $model_path
        python finetune.py \
    	--num_workers 2 \
    	--split "scaffold" \
    	--gnn_type gin \
    	--dataset $dataset_name \
    	--dataset_path $dataset_path \
    	--lr 1e-3 \
    	--epochs $num_epochs \
	--runseed $seed \
    	--input_model_file $model_path 
      done
    done
   done
done
