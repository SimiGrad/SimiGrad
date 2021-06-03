from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

class FalseDict(object):
    def __getitem__(self,key):
        return 0
    def __contains__(self, key):
        return True

adaptive_batch_results=[]
pandas_results=pd.DataFrame(columns=["name","average_batch_size","acc"])
pandas_results.set_index("name",inplace=True)
print(pandas_results)
print(pandas_results.index)
try:
    cache=np.load("full_cache.npy", allow_pickle=True).item()
except FileNotFoundError:
    cache={}
for root,dirs,files in os.walk("log"):
    for dir in dirs:
        if dir not in cache:
            filename=os.path.join(root,dir)
            event_acc = EventAccumulator(filename,size_guidance=FalseDict())
            event_acc.Reload()
            w_times, step_nums, vals = zip(*event_acc.Scalars("val_top1_acc")) #get total steps
            acc=max(vals)
            w_times, step_nums, vals = zip(*event_acc.Scalars("number_of_samples_vs_update_steps")) #get total number of samples
            total_samples=vals[-1]
            number_of_steps=step_nums[-1]
            average_batch_size=total_samples/number_of_steps
            cache[dir]=(average_batch_size,acc)
        else:
            average_batch_size,acc=cache[dir]
        adaptive_batch_results.append((dir,average_batch_size,acc))
np.save("full_cache",cache)

def remove_nonconvex_point(results):
    for idx,point in enumerate(results):
        for other_point in results:
            if point[1]<=other_point[1] and point[2]<other_point[2]:
                del results[idx]
                return results,False
    return results,True

def make_strict_convex_hull(results):
    while True:
        results,complete=remove_nonconvex_point(results)
        if complete:
            break
    # while True:
    #     results,complete=remove_nonconvex_point_strict(results)
    #     if complete:
    #         break
    return results

make_strict_convex_hull(adaptive_batch_results)

adaptive_batch_results.sort(key=lambda x:x[2])

for result in reversed(adaptive_batch_results):
    print("\t".join([str(n) for n in result]))

    pandas_results.loc[result[0]]=[result[1],result[2]]
pandas_results.to_csv("imagenet_pareto.csv")