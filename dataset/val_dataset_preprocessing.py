import shutil
import os

for i, line in enumerate(map(lambda s: s.strip(), open("./val/val_annotations.txt"))):
    name, wnid = line.split("\t")[0:2]  
    origin_path_file = "./val/images/" + name
    destination_path = "./val/" + wnid
    destination_path_file = "./val/" + wnid + "/" + name
    if not os.path.exists(destination_path): os.makedirs(destination_path)
    shutil.move(origin_path_file, destination_path_file)
    print(destination_path_file)
    
os.rmdir("./val/images")  
print("Done!")