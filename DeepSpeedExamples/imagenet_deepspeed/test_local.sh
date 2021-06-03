{
pkill python -u $UID
sleep 10s
mkdir -p log

name=resnet50_baseline_seed1
deepspeed --hostfile /dev/null main_deepspeed.py --arch resnet50 --seed 1 --epochs 90 --mixup 0.0 --workspace ./ --name $name --deepspeed --deepspeed_config config_bsz2k.json |& tee log/$name.log
exit
}