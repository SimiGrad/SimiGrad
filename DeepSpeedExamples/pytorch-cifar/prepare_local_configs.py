import json
import sys

micro_batch=sys.argv[1]
step=sys.argv[2]

with open("ds_config_tmp.json") as f:
    a=json.load(f)
    a["train_micro_batch_size_per_gpu"]=int(micro_batch)
    a["gradient_accumulation_steps"]=int(step)
print(a)
json.dump(a, open("ds_config_tmp.json","w"),indent=1)

