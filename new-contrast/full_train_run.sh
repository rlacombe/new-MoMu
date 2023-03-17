eps=(0.5)
temps=(0.2 0.1 0.05)
sampling_methods=("cos_sim_mean" "cos_sim_max" "cos_sim_sent")

for sampling_method in "${sampling_methods[@]}"
do
  for temp in "${temps[@]}"
  do
    echo $temp
    echo $sampling_method
    python train_gin.py --root='../data/contrast-pretrain/XL/' --batch_size=128 --accelerator='gpu' --gpus='1' --graph_self --max_epochs=2 --num_workers=1 --log_every_n_steps=10 --sampling_type=$sampling_method --sampling_temp=$temp --sampling_eps=0.5;
  done
done

