import os, shutil
from modal import Volume

v = Volume.from_name("dataset")
v.mount("/data")

path = "/data/plots"

if os.path.exists(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        print("Deleted directory:", path)
    else:
        os.remove(path)
        print("Deleted file:", path)
else:
    print("Not found:", path)
