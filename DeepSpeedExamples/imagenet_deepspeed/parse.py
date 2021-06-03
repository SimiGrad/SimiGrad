import re
import numpy as np
import os

result=[]
for line in open(os.path.join("log","original.log")):
    if "train.loss" in line and "val.top1" in line:
        t=line.split()
        for idx,piece in enumerate(t):
            if piece=="val.top1":
                result.append(float(t[idx+2]))
result=[result[n] for n in range(0,len(result),16)]
print(result)
np.save("result",result)
np.savetxt("result.csv",result,delimiter=",")
