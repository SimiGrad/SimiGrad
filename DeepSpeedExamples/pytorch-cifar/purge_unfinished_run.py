import os
import shutil

for root, dirs, files in os.walk("log", topdown=False):
    for name in dirs:
        if not os.path.exists(os.path.join("log",name+".txt")):
            print(name)
            shutil.rmtree(os.path.join("log",name))

            