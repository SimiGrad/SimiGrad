import os
import shutil
import time

for root,dirs,files in os.walk("outputs"):
    for dir in dirs:
        if not os.path.exists(os.path.join(root,dir,"predictions.json")):
            input(f"{dir} REMOVE?")
            os.remove(os.path.join("log",f"{dir}.log"))
            # shutil.rmtree(os.path.join(root,file.replace(".log","")))

