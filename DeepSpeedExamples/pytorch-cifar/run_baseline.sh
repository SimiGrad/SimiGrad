{
pkill python -u $UID
sleep 20s

similirity_target=0.2
bszbound=4k

name=resnet18_bsz128_100epoch_xavier_normal_adabatchlr

mkdir -p log

deepspeed main_xavier_normal_adabatchlr.py --name $name --seed 1  --deepspeed --deepspeed_config ds_config_flexible.json |& tee -a log/$name.txt

exit
}