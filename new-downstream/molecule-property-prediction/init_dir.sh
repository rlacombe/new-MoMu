# Script for initializing the directory so that it's ready to run molecule generation

# Download the data.
dataset_names=(muv hiv bace bbbp tox21 toxcast sider clintox)
dataset_paths=(muv HIV bace BBBP tox21 toxcast_data sider clintox) 
zipped=(true false true false false true true true true)
for (( i=0; i<${#dataset_names[@]}; i++ ));
do
  # download dataset
  dataset_file_name=${dataset_paths[i]}.csv
  if [ ${zipped[i]} = true ]
  then
    dataset_file_name=${dataset_paths[i]}.csv.gz
  fi
  wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/$dataset_file_name

  # Put dataset in correct directory
  mkdir -p dataset/${dataset_names[i]}/raw
  mv $dataset_file_name dataset/${dataset_names[i]}/raw

  # Unzip and do everything else
  cd dataset/${dataset_names[i]}/raw
  if [ ${zipped[i]} ]
  then
    gunzip -f $dataset_file_name 
    dataset_file_name=${dataset_paths[i]}.csv
  fi
  mv $dataset_file_name ${dataset_names[i]}.csv
  cd ../../../
done

# Download the checkpoints
# TODO
