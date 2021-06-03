import os
import subprocess
# import numpy as np


# similarity_range=[0,1]
# bszbound_range=[8,15]

# interval=0
# while True:
#     interval+=1
#     for i in range(interval+1):
#         for j in range(interval+1):
#             similarity=similarity_range[0]+i/interval*(similarity_range[1]-similarity_range[0])
#             bszbound=int(2**(bszbound_range[0]+j/interval*(bszbound_range[1]-bszbound_range[0])))
#             if not os.path.exists(os.path.join("log",f"resnet18_bsz128_target{similarity}_bszbound{bszbound}")):
#                 print("similarity",similarity,"bszbound",bszbound)
#                 subprocess.call(["./run_with_arguments.sh",str(similarity),str(bszbound)])#, shell=True

bszboundk=32
while bszboundk>1:
    bszboundk=bszboundk//2
    for lower_bound in reversed([32,64,128,256,512,1024]):
        if lower_bound>=bszboundk*1024:
            continue
        for similarity in range(0,21):
            similarity=similarity/100
        # for similarity in [0.01,0.05]:
            bszbound=str(bszboundk)+"k"
            if not os.path.exists(os.path.join("xavier_normal_log",f"resnet18_bsz128_target{similarity}_bszbound{lower_bound}_{bszbound}")):
                print("similarity",similarity,"lower_bound",lower_bound,"bszbound",bszbound)
                subprocess.call(["./run_with_min_arguments.sh",str(similarity),str(bszbound),str(lower_bound)])#, shell=True

            