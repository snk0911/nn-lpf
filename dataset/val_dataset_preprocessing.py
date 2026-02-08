import shutil
import os

for i, line in enumerate(map(lambda s: s.strip(), open("./tiny-imagenet-200/val/val_annotations.txt"))):
    name, wnid = line.split("\t")[0:2]  
    origin_path_file = "./tiny-imagenet-200/val/images/" + name
    destination_path = "./tiny-imagenet-200/val/" + wnid
    destination_path_file = "./tiny-imagenet-200/val/" + wnid + "/" + name
    if not os.path.exists(destination_path): os.makedirs(destination_path)
    shutil.move(origin_path_file, destination_path_file)
    print(destination_path_file)
    
os.rmdir("./tiny-imagenet-200/val/images")
print("Done!")
