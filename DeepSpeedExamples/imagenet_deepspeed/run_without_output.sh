{
pkill python -u $UID
sleep 10s
mkdir -p log

similarity_target=$1
bszbound=$2
bsz_lowerbound=$3

name=resnet50_bsz2k_target${similarity_target}_bszbound${bsz_lowerbound}_${bszbound}
deepspeed --hostfile /dev/null main_deepspeed.py --arch resnet50 --epochs 90 --mixup 0.0 --name $name --similarity_target ${similarity_target} --batchsize_lower_bound ${bsz_lowerbound} --batchsize_upper_bound ${bszbound} --deepspeed --deepspeed_config config_bsz2k.json |& tee log/$name.log
exit
}