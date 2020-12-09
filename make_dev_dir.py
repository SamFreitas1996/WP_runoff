import os
import random 
import numpy as np
from natsort import natsorted
import pathlib


data_dir2 = pathlib.Path(os.path.join(os.getcwd(),'TFWP_training'))

print(data_dir2)

w_ex_path = os.path.join(os.getcwd(),'TFWP_training','worms')
nw_ex_path = os.path.join(os.getcwd(),'TFWP_training','no_worms')

# os.replace("path/to/current/file.foo", "path/to/new/destination/for/file.foo")

all_w_files = natsorted(os.listdir(w_ex_path))
all_w_images_array = []
for each_file in all_w_files:
    if(each_file.endswith(".jpg") or each_file.endswith(".png")):
        all_w_images_array.append(os.path.join(w_ex_path,each_file))

all_nw_files = natsorted(os.listdir(nw_ex_path))
all_nw_images_array = []
for each_file in all_nw_files:
    if(each_file.endswith(".jpg") or each_file.endswith(".png")):
        all_nw_images_array.append(os.path.join(nw_ex_path,each_file))


dev_w_list = random.sample(all_w_images_array, 100)
dev_nw_list = random.sample(all_nw_images_array, 100)

# os.mkdir(os.path.join(os.getcwd(),'TFWP_dev'))
# os.mkdir(os.path.join(os.getcwd(),'TFWP_dev','worms'))
# os.mkdir(os.path.join(os.getcwd(),'TFWP_dev','no_worms'))

for count,each_file in enumerate(dev_w_list):
    os.replace(each_file,os.path.join(os.getcwd(),'TFWP_dev','worms',str(count)+'.png'))
    # os.replace("path/to/current/file.foo", "path/to/new/destination/for/file.foo")

for count,each_file in enumerate(dev_nw_list):
    os.replace(each_file,os.path.join(os.getcwd(),'TFWP_dev','no_worms',str(count)+'.png'))
    # os.replace("path/to/current/file.foo", "path/to/new/destination/for/file.foo")

print('You are a bold on')
