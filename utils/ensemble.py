""" 
# This script creates a ensemble submission using all submissions from other models inside the SUBMISSION_PATH folder
"""

from PIL import Image
import glob
import io
import os
import matplotlib.pyplot as plt
from skimage.io import imsave
import numpy as np
from tqdm import tqdm
import pandas as pd

SUBMISSIONS_PATH = "../submissions/" 

## ------------------------------------------------------------------ UTIL FUNCTIONS START -----------------------------------------------------------

def create_path_list(dir):
    """ We make a list of all the paths in the given folder """

    paths = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname) 
            paths.append(path)

    return paths

## ------------------------------------------------------------------ UTIL FUNCTIONS END -----------------------------------------------------------

submissions_path = IMAGES_TO_NORMALIZE = sorted(create_path_list(SUBMISSIONS_PATH))

submission_pd = []
for path in submissions_path:
    submission_pd.append(pd.read_csv(path, header=0, index_col=0))

res = pd.DataFrame(index=submission_pd[0].index, columns=['prediction'])

print(res)

indx = 0
for _, row in tqdm(res.iterrows()):
    count = 0
    for submission in submission_pd:
        count += submission.iloc[indx, 0]
    
    if count > len(submission_pd)/2:
        res.iloc[indx, 0] = 1
    else:
        res.iloc[indx, 0] = 0
    indx += 1

res.to_csv("ensemble_submission.csv")