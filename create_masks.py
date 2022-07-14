# Imports PIL module 
import os
from PIL import Image
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from pathlib import Path    

def main():
    directory = "results/roadmaps"
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):

            roadmap = Image.open(f) 
            roadmap = np.array(roadmap.convert('RGB'))
            mask = (np.floor(rgb2gray(np.floor(roadmap / 255)))*255).astype(np.uint8)
            mask = np.expand_dims(mask, axis=2)
            mask = np.repeat(mask, 3, axis=2)
            mask = Image.fromarray(mask)


            f_name = Path(f).name
            mask.save("result/masks/" + f_name)


if __name__ == "__main__":
    main()
