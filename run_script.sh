#!/bin/bash
# A bash script to run MR reconstruction or training without typing all the commands

source /home/tjao/miniconda3/bin/activate
conda activate dit-cs
#conda activate latte-cs

recon_train_flag=0 # 0 for training, 1 for reconstruction
num_emaps=2  #1 or 2 emaps
batch_recon=1 #0 recon individual files, 1 recon folder of files
recon_ckpt=$1

root_dir="/home/tjao/data/"

#out_directory=$root_dir"recon_${num_emaps}emaps/"
out_directory="/home/tjao/data/recon_${num_emaps}emaps_val_set/"

#data_directory=$root_dir"stanfordCine_${num_emaps}emaps/test"
data_directory="/home/tjao/data/stanfordCine_${num_emaps}emaps/validate"

#ckpt_directory=$root_dir"stanfordCine_${num_emaps}emaps/summary/train-3D_5steps_2SWINblocks_80features_2emaps_0weight_6SWIN_788WIN/"
#ckpt_directory=$root_dir"stanfordCine_${num_emaps}emaps/summary/example/train-3D_5steps_2resblocks_64features_2emaps_0weight/"
#ckpt_directory=$root_dir"stanfordCine_${num_emaps}emaps/summary/train-3D_5steps_4RESblocks_180features_2emaps_0weight/"

#ckpt_directory=$root_dir"stanfordCine_${num_emaps}emaps/DiT/DDPM_E_1steps_24DiTblock_16Heads_384features_FS/"
ckpt_directory=$root_dir"stanfordCine_${num_emaps}emaps/Latte/DDPM_X_1steps_12DiTblock_6Heads_192features_4Patch_FS/"


model_type="Latte" #RES or SE or CBAM or SWIN or DIT or Latte
acceleration=1

#Setting the config file
if [[ "$model_type" == "SWIN" ]]
then
  config_file="/home/tjao/code/dl-cs-dynamic/configs/config_swin.yaml"
else 
  if [[ "$model_type" == "DIT" ]]
  then
    config_file="/home/tjao/code/dl-diff/configs/config_dit.yaml"
  else 
    if [[ "$model_type" == "Latte" ]]
    then
      config_file="/home/tjao/code/dl-diff/configs/config_latte.yaml"
    else
      config_file="/home/tjao/code/dl-cs-dynamic/configs/config_se.yaml"
    fi
  fi
fi 

echo $config_file

#Resuming from checkpoint
if [[ $recon_ckpt -eq 1 ]]
then
  echo $ckpt_directory
  ckpt_file=$(ls -td $ckpt_directory*.ckpt | head -n 1)
  echo $ckpt_file

  epoch_num="$(cut -d'=' -f2 <<<$ckpt_file)"
  epoch_num="$(cut -d'-' -f1 <<<$epoch_num)"

  echo "Epoch Number is "$epoch_num
fi
#Only resume from checkpoint if epoch is <950
if [[ $(($epoch_num)) -le 950 ]]
then

  # Training
  if [[ $recon_train_flag -eq 0 ]]
  then
    echo "Perform Training"

    #SE Training
    if [[ "$model_type" == "SE" ]]
    then
      echo "Squeeze Excitation"
      python3 scripts/train_se.py --config-file $config_file --device 0
    fi

    #RESNET Training
    if [[ "$model_type" == "RES" ]]
    then
      echo "RESNET"

      if [[ $recon_ckpt -eq 0 ]]
      then
        python3 scripts/train.py --config-file $config_file --device 0
      else
        python3 scripts/train.py --config-file $config_file --device 0 --resume --ckpt $ckpt_file
      fi
    fi

    #CBAM Training
    if [[ "$model_type" == "CBAM" ]]
    then
      echo "CBAM"
      python3 scripts/train_cbam.py --config-file $config_file --device 0
    fi

    #SWIN Training
    if [[ "$model_type" == "SWIN" ]]
    then
      echo "SWIN"

      if [[ $recon_ckpt -eq 0 ]]
      then
        python3 scripts/train_swin.py --config-file $config_file --device 0
      else
        python3 scripts/train_swin.py --config-file $config_file --device 0 --resume --ckpt $ckpt_file
      fi
    fi

    #DiT Training
    if [[ "$model_type" == "DIT" ]]
    then
      echo "DIT"

      if [[ $recon_ckpt -eq 0 ]]
      then
        python3 scripts/train_DiT.py --config-file $config_file --device 0
      else
        python3 scripts/train_DiT.py --config-file $config_file --device 0 --resume --ckpt $ckpt_file
      fi
    fi

    #Latte Training
    if [[ "$model_type" == "Latte" ]]
    then
      echo "Latte"

      if [[ $recon_ckpt -eq 0 ]]
      then
        python3 scripts/train_Latte.py --config-file $config_file --device 0
      else
        python3 scripts/train_Latte.py --config-file $config_file --device 0 --resume --ckpt $ckpt_file
      fi
    fi

  fi

  #Resume Training
  #ckpt_file="/home/tjao/data/stanfordCine_${num_emaps}emaps/summary/example/train-3D_5steps_2CBAMblocks_64features_2emaps_2weight/epoch\=155-step\=40715.ckpt"
  #python3 scripts/train_cbam.py --config-file $config_file --device 0 --resume --ckpt $ckpt_file

  #ckpt_file="/home/tjao/data/stanfordCine_2emaps/summary/example/train-3D_5steps_2SEblocks_64features_2emaps_2weight/epoch\=503-step\=131543.ckpt"
  #python3 scripts/train_se.py --config-file $config_file --device 0 --resume --ckpt $ckpt_file

fi

#Reconstruction
if [[ $recon_train_flag -eq 1 ]]
then
  test_file="/home/tjao/data/cfl_${num_emaps}emaps/"
  # RES with VGG weight
  # ckpt_file="/home/tjao/data/stanfordCine_${num_emaps}emaps/summary/example/train-3D_5steps_2RESblocks_64features_2emaps_2weight/epoch\=891-step\=232811.ckpt"
  # SE with VGG weight
  # ckpt_file="/home/tjao/data/stanfordCine_${num_emaps}emaps/summary/example/train-3D_5steps_2SEblocks_64features_2emaps_2weight/epoch\=985-step\=257345.ckpt"
  # SE with no weight
  #ckpt_file="/home/tjao/data/stanfordCine_${num_emaps}emaps/summary/example/train-3D_5steps_2SEblocks_64features_2emaps_0weight/epoch\=992-step\=259172.ckpt"
  # RES with no weight
  # ckpt_file="/home/tjao/data/stanfordCine_${num_emaps}emaps/summary/example/train-3D_5steps_2resblocks_64features_2emaps_0weight/epoch\=983-step\=256823.ckpt"

  #ckpt_file="/home/tjao/data/stanfordCine_${num_emaps}emaps/summary/example/train-3D_5steps_2SEblocks_64features_1emaps_0weight/epoch\=990-step\=258650.ckpt"
  #ckpt_file="/home/tjao/data/stanfordCine_${num_emaps}emaps/summary/example/train-3D_5steps_2resblocks_64features_1emaps_0weight/epoch\=761-step\=198881.ckpt"
  #ckpt_file="/home/tjao/data/stanfordCine_${num_emaps}emaps/summary/example/train-3D_5steps_2SEblocks_64features_2emaps_1weight/epoch\=984-step\=257084.ckpt"
  #ckpt_file="/home/tjao/data/stanfordCine_${num_emaps}emaps/summary/example/train-3D_5steps_2CBAMblocks_64features_2emaps_0weight_v2/epoch\=917-step\=239597.ckpt"
  #ckpt_file="/home/tjao/data/stanfordCine_${num_emaps}emaps/summary/example/train-3D_5steps_2CBAMblocks_64features_2emaps_2weight/epoch\=155-step\=40715.ckpt"

  if [[ $batch_recon -eq 0 ]]
  then
    echo "Perform Single Image Reconstruction"
    python3 scripts/reconstruct_h5.py --config-file $config_file --device 0 --ckpt $ckpt_file --directory $test_file --model $model_type
  else
    echo "Perform Batch Reconstruction"
    echo $config_file
    echo $ckpt_file
    python3 batch_recon.py --config-file $config_file --acceleration $acceleration --out-directory $out_directory --device 0 --ckpt $ckpt_file --data-dir $data_directory --model $model_type
  fi
fi
