import os
import subprocess
import math


bszbound=64*1024
while bszbound>128:
    bszbound=bszbound//2
    # lr=0.1*math.sqrt(bszbound//128)
    lr=0.1*(bszbound//128)
    if bszbound%1024==0:
        bszbound_str=f"{bszbound//1024}k"
    else:
        bszbound_str=str(bszbound)
    if not os.path.exists(os.path.join("log",f"resnet18_bsz{bszbound_str}_lr{'%.2f' % lr}_warmup")):
        # print("similarity",similarity,"bszbound",bszbound)
        if bszbound>256*4:
            microbsz=256
            gradient_accumulatiom_steps=bszbound//microbsz//4
        else:
            microbsz=bszbound//4
            gradient_accumulatiom_steps=1
        subprocess.call(["python","prepare_local_configs.py",str(microbsz),str(gradient_accumulatiom_steps)])#, shell=True
        subprocess.call(["./run_with_arguments.sh",str(bszbound_str),'%.2f' % lr])#, shell=True
        for line in open(os.path.join("log",f"resnet18_bsz{bszbound_str}_lr{'%.2f' % lr}_warmup.txt")):
            if "Exception" in line or "Error" in line:
                print(line)
                raise Exception