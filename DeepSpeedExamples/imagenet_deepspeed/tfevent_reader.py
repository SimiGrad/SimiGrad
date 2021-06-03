from numpy.core import einsumfunc
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
# import matplotlib.pyplot as plt

class FalseDict(object):
    def __getitem__(self,key):
        return 0
    def __contains__(self, key):
        return True

results=[]
for root,dirs,files in os.walk("log"):
    for dir in dirs:
        filename=os.path.join(root,dir)
        if "resnet50_baseline_seed1" not in filename:
            continue
        # print(filename)

        event_acc = EventAccumulator(filename,size_guidance=FalseDict())
        event_acc.Reload()
        # Show all tags in the log file
        # print(event_acc.Tags())

        _, _, val_top1_acc = zip(*event_acc.Scalars("val_top1_acc"))
        _, _, adjusted_batch_size = zip(*event_acc.Scalars("adjusted_batch_size"))
        # print(dir,"\t",len(adjusted_batch_size),max(val_top1_acc))
        # if len(val_top1_acc)>=149:
        results.append((dir,len(adjusted_batch_size),max(val_top1_acc)))

results.sort(key=lambda x:x[2])

for result in reversed(results):
    print("\t".join([str(n) for n in result]))
