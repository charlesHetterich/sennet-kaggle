# handles additional setup stuff from a python script (w.r.t setup.sh in root directory)

import os
from glob import glob

data_dir = "/root/data"
k3d_dir = f"{data_dir}/train/kidney_3_dense"
os.mkdir(f"{k3d_dir}/images")

k3d_slices = glob(f"{k3d_dir}/labels/*")
k3d_slices = [os.path.basename(x) for x in k3d_slices]
slices_to_copy = [
    pth for pth 
    in glob(f"{data_dir}/train/kidney_3_sparse/images/*") 
    if os.path.basename(pth) in k3d_slices
]

for pth in slices_to_copy:
    os.system(f"cp {pth} {k3d_dir}/images")