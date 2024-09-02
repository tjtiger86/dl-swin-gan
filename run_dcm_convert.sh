#run_im_matrix

#!/bin/bash
# A bash script to run MR reconstruction or training without typing all the commands

source /home/tjao/miniconda3/bin/activate
conda activate dl-cs

code_dir="/home/tjao/code/dl-cs-dynamic"
root_dir="/home/tjao/data/"
recon_dir="recon_2emaps/"
#recon_dir="recon_2emaps_val_set/"
save_dir="/home/tjao/Documents/SWIN_RES_COMPARE/dcm_7792_8/"
#recon_dir="recon_2emaps/"

recon_type="resnet" #resnet vs. swin

if [[ "$recon_type" == "resnet" ]]
then
    im_dir=$root_dir$recon_dir"train-3D_5steps_2resblocks_64features_2emaps_0weight/"
else 
    im_dir=$root_dir$recon_dir"train-3D_5steps_1SWINblocks_160features_2emaps_0weight_6SWIN_788WIN/"
fi

#im="Exam5050_Series12_"

#im="Exam2200_Series8_"
im="Exam7792_Series8_"
#im="Exam9966_Series12_"

declare -a accel=( 
    [0]="1"
    [1]="12"
    [2]="16"
    [3]="20"
    [4]="24"
    )

declare -a fname=()

declare -a xrange=( 
    [0]="52"
    [1]="179"
    )

declare -a yrange=( 
    #Exam5050_Series12
    [0]="17"
    [1]="144"
    )

declare -a im_expand=(
    [0]="256"
    [1]="256"
)

echo $save_file

count=0
for i in ${accel[@]}
do 
fname[$count]=$im_dir$im$i"accel.im"
count=$count+1
done 


echo "${accel[@]}"
echo "${fname[@]}"

#python3 $code_dir/write_dcm.py --file ${fname[@]} --ext .dcm --save_dir $save_dir$recon_type"_"
#python3 $code_dir/write_dcm.py --file ${fname[@]} --ext .dcm --save_dir $save_dir --study_type $recon_type

python3 $code_dir/write_dcm.py --file ${fname[@]} --ext .dcm --save_dir $save_dir --study_type $recon_type --xrange ${xrange[@]} --yrange ${yrange[@]} --im_size ${im_expand[@]}

