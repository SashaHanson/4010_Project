import os, shutil
from modal import Volume


#modal does not allow users to delete stuff directly from the volume, so we need to run a script
#to be able to delete this stuff
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
