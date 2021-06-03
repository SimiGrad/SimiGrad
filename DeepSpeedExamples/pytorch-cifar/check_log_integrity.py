import os

for root,dirs,names in os.walk("log"):
    for dir in dirs:
        if len(os.listdir(os.path.join(root,dir)))>1:
            print(dir)
