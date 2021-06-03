import os
import subprocess
import signal
import nvsmi
import time

def sig_exit(sig,stack):
    print("Got SIGTERM, ignored.")
signal.signal(signal.SIGTERM, sig_exit)

if (list(nvsmi.get_gpus())[0].mem_total)<20000:
    flag_low_end_gpu=True
else:
    flag_low_end_gpu=False

while True:
    for root,dirs,files in os.walk("./bert_model_outputs/saved_models"):
        for run in dirs:
            print(f"checking {run}")
            ckpts=os.listdir(os.path.join(root,run))
            run_finished=False
            for ckpt in ckpts:
                if "epoch150" in ckpt:
                    run_finished=True
            if run_finished and "seq512" not in run and "bert_large" in run and not os.path.exists(os.path.join("./bert_model_outputs",f"{run}_seq512_170epoch.log")):
                print(run)
                subprocess.call(["./ds_train_bert_bsz32k_seq512.sh",str(run)])#, shell=True
    time.sleep(60)   