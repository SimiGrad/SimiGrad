import json
import sys

batch_size=sys.argv[1]

with open("tmp_config.json") as f:
    a=json.load(f)
    a["train_batch_size"]=int(batch_size)
print(a)
json.dump(a, open("tmp_config.json","w"),indent=1)

