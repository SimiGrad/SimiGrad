{
pkill python -u $UID
sleep 20s

similirity_target=0.2
bszbound=4k

name=resnet18_bsz128_target${similirity_target}_bszbound${bszbound}

mkdir -p log

deepspeed main.py --name $name --similirity_target ${similirity_target} --batchsize_upper_bound ${bszbound} --seed 1  --deepspeed --deepspeed_config ds_config_flexible.json |& tee -a log/$name.txt

exit
}