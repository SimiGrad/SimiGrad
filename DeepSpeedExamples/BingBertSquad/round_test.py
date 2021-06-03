import os
import subprocess
# from natsort import natsort_key
from collections import defaultdict
import time
import os
import signal
import psutil
import sys
import torch
import nvsmi
import apex
import deepspeed
import boto3

assert deepspeed.git_version_info.git_hash=="fd2a8fd"

ckpt_dir=

gpu_number=torch.cuda.device_count()
if (list(nvsmi.get_gpus())[0].mem_total)<20000:
    flag_low_end_gpu=True
else:
    flag_low_end_gpu=False
assert 24%gpu_number==0
if flag_low_end_gpu:
    local_bsz=1
if not flag_low_end_gpu:
    local_bsz=3


def sigint_handler(signal, frame):
    print("Kill all subprocess")
    procs = psutil.Process().children(recursive=True)
    for p in procs:
        p.kill()
    sys.exit(0)

def sigterm_handler(signal, frame):
    print("Got SIGTERM, ignore")

signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGTERM, sigterm_handler)

for repeat_test in range(32):
    for epoch_to_test in range(170,165,-1):
        for root, dirs, files in os.walk(ckpt_dir):
            # dirs.sort(key=natsort_key, reverse=True)
            for name in files:
                # print(name)
                if name.endswith(".pt"):
                    ckpt_name=root.split("/")[-2]
                    ckpt_epoch=root.split("/")[-1].split("_")[0]
                    assert ckpt_epoch.startswith("epoch")
                    if ckpt_epoch==f"epoch{epoch_to_test}":
                        print(ckpt_name,ckpt_epoch,repeat_test)
                        parameters=["./run_squad_deepspeed.sh",gpu_number,f"{ckpt_name}_{ckpt_epoch}",os.path.join(root,name),repeat_test,local_bsz]
                        subprocess.call([str(item) for item in parameters])#, shell=True

