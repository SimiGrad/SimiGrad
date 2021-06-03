from numpy.lib.function_base import average
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import numpy as np
import matplotlib.pyplot as plt

class FalseDict(object):
    def __getitem__(self,key):
        return 0
    def __contains__(self, key):
        return True

adaptive_batch_results=[]
try:
    cache=np.load("cache.npy", allow_pickle=True).item()
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
        adaptive_batch_results.append((average_batch_size,acc))
np.save("cache",cache)

# results.sort(key=lambda x:x[2])
adaptive_batch_results.sort()

# for result in reversed(results):
#     print("\t".join([str(n) for n in result]))

def remove_nonconvex_point_strict(results):
    for idx,point in enumerate(results):
        if idx<len(results)-2:
            if results[idx+1][0]==results[idx][0]:
                if results[idx+1][1]>results[idx][1]:
                    del results[idx]
                else:
                    del results[idx+1]
            slope1=(results[idx+1][1]-results[idx][1])/(results[idx+1][0]-results[idx][0])
            slope2=(results[idx+2][1]-results[idx][1])/(results[idx+2][0]-results[idx][0])
            if np.abs(slope1)<np.abs(slope2)*0.8:
                del results[idx+1]
                return results,False
    return results,True

def remove_nonconvex_point(results):
    for idx,point in enumerate(results):
        for other_point in results:
            if point[0]<other_point[0] and point[1]<other_point[1]:
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

adaptive_batch_results=make_strict_convex_hull(adaptive_batch_results)
x,y=zip(*adaptive_batch_results)
plt.plot(x,y,'-o', label="our method")

linear_lr_results=[]
sqrt_lr_results=[]
for name in cache:
    if "target" not in name and "baseline" not in name:
        try:
            bsz=name.split("_")[1].replace("bsz","")
            if bsz.endswith("k"):
                bsz=int(bsz.replace("k",""))*1024
            else:
                bsz=int(bsz)
            lr=float(name.split("_")[2].replace("lr",""))
        except:
            continue
        linear_lr=0.1*bsz/2048
        sqrt_lr=0.1*np.sqrt(bsz/2048)
        if np.abs(lr-linear_lr)<np.abs(lr-sqrt_lr):
            linear_lr_results.append(cache[name])
        else:
            sqrt_lr_results.append(cache[name])
    if "seed1" in name:
        linear_lr_results.append(cache[name])
        sqrt_lr_results.append(cache[name])

def plot(results,line_style='-o',label=None):
    results=sorted(results)
    x,y=zip(*results[:-1])
    plt.plot(x,y,line_style,label=label)

plot(linear_lr_results,label="linear learning rate baseline (warmup)")
plot(sqrt_lr_results,label="sqrt learning rate baseline (warmup)")




plt.legend()
plt.xlabel("Average Batch Size")
plt.ylabel("Test Acc. (%)")

plt.savefig("Pareto.png")


plt.ylim(bottom=60)
plt.xlim(left=120,right=75*1024)

plt.savefig("Pareto_zoomed1.png")

plt.ylim(bottom=60)
plt.xlim(left=120,right=18*1024)

plt.savefig("Pareto_zoomed2.png")
