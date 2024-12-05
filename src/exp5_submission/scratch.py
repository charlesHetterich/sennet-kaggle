import torch
from monai.metrics import SurfaceDiceMetric
import pandas as pd
import numpy as np

submission_df = pd.read_csv('submission.csv')

def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

print("loading data...")
pmask = torch.stack(
    [torch.tensor(rle_decode(row.rle, (1041, 1511)))
    for row in submission_df.itertuples()]
).unsqueeze(0).bool()
true_mask = torch.load('/root/data/cache/train/kidney_2/labels.pt')

surface_dice_metric = SurfaceDiceMetric([0.5], include_background=False, distance_metric='euclidean')
print("calculating surface dice...")
print(surface_dice_metric(pmask.unsqueeze(0), true_mask.unsqueeze(0)))