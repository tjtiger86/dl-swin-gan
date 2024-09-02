#run_im_matrix

#!/bin/bash
# A bash script to run MR reconstruction or training without typing all the commands

source /home/tjao/miniconda3/bin/activate
conda activate dl-cs

code_dir="/home/tjao/code/dl-cs-dynamic"
root_dir="/home/tjao/data/"
#recon_dir="recon_2emaps_val_set/"
recon_dir="recon_2emaps/"
res_im_dir=$root_dir$recon_dir"train-3D_5steps_2resblocks_64features_2emaps_0weight/"
swin_im_dir=$root_dir$recon_dir"train-3D_5steps_1SWINblocks_160features_2emaps_0weight_6SWIN_788WIN/"

#im="Exam5050_Series12_"
#slice=3

#im="Exam2200_Series8_"
#slice=0

#im="Exam7792_Series8_"
#slice=3

im="Exam9966_Series12_"
slice=0

animate=0
phase=10

declare -a accel=( 
    [0]="12"
    [1]="16"
    [2]="20"
    [3]="24"
    )

declare -a xrange=( 
    #Exam5050_Series12
    [0]="58"
    [1]="138"

    #Exam2200_Series8
    #[0]="70"
    #[1]="150" 

    #"Exam7792_Series8_"
    #[0]="65"
    #[1]="165"

    #[0]="0"
    #[1]="-1"

    #"Exam9966_Series12_"
    [0]="40"
    [1]="145"

    )

declare -a yrange=( 
    #Exam5050_Series12
    [0]="20"
    [1]="100"
    
    #Exam2200_Series8
    #[0]="70"
    #[1]="150" 

    #"Exam7792_Series8_"
    #[0]="30"
    #[1]="130"

    #[0]="0"
    #[1]="-1"

    #"Exam9966_Series12_"
    [0]="35"
    [1]="130"

    )

declare -a fname=()
declare -a dname=()

declare -a matrix=(
    [0]="2"
    [1]="$(( ${#accel[@]}+1 ))"
)

if [[ $(( $animate )) -eq 0 ]]
then 
    ext=.jpg
else 
    ext=.mp4
fi 

save_dir="/home/tjao/Documents/SWIN_RES_COMPARE/"
outfile=$save_dir$im"accel12_16_20_24"$ext
outfile_error=$save_dir$im"accel12_16_20_24Error"$ext

echo $save_file

count=0
for i in ${accel[@]}
do 
fname[$count]=$res_im_dir$im$i"accel.im"
fname[$count+1]=$swin_im_dir$im$i"accel.im"
count=$count+2
done 

#Comparison with Acceleration of 1, ground truth
fname[$count+1]=$swin_im_dir$im"1accel.im"

for i in ${accel[@]}
do 
dname[$count]=$swin_im_dir$im"1accel.im"
dname[$count+1]=$swin_im_dir$im"1accel.im"
count=$count+2
done 


echo "${accel[@]}"
#echo "${fname[@]}"
#echo "${matrix[@]}"

python3 $code_dir/display_matrix.py --file ${fname[@]} --matrix ${matrix[@]} --animate $animate --slice $slice --phase $phase --outfile $outfile --scale -1 3 --xrange ${xrange[@]} --yrange ${yrange[@]} 

unset fname[${#fname[@]}] #remove the extra Ground truth image

python3 $code_dir/display_matrix.py --file ${fname[@]} --dfile ${dname[@]} --matrix ${matrix[@]} --animate $animate --slice $slice --phase $phase --outfile $outfile_error --cmap viridis --scale -1 -1 --xrange ${xrange[@]} --yrange ${yrange[@]} 