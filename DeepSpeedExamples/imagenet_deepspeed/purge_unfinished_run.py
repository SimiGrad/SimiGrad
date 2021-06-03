import os
import shutil
import time

for root,dirs,files in os.walk("log"):
    for file in files:
        if not file.endswith(".log"):
            continue
        if time.time()-os.path.getmtime(os.path.join(root,file))<60*180:
            continue
        flag_finished=False
        for line in reversed(list(open(os.path.join(root,file)))):
            if "Experiment ended" in line:
                flag_finished=True
        if not flag_finished:
            print(file)
            os.remove(os.path.join(root,file))
            shutil.rmtree(os.path.join(root,file.replace(".log","")))

