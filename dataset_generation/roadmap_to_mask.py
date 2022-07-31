"""
#
# This script creates masked images from roadmap images using a thresholding algorithm
#  
"""

# Imports PIL module 
import os
from PIL import Image
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from pathlib import Path    

def main():
    # open method used to open different extension image file
    for i in range(0,20):
        directory = "data/google/roadmaps/{0}".format(str(i))
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(f):
                path = f
                roadmap = Image.open(f) 
                roadmap = np.array(roadmap.convert('RGB'))
                mask = (np.floor(rgb2gray(np.floor(roadmap / 255)))*255).astype(np.uint8)
                mask = np.expand_dims(mask, axis=2)
                mask = np.repeat(mask, 3, axis=2)
                mask = Image.fromarray(mask)

                path = "data/google/masks/{0}".format(str(i))
                if not os.path.exists(path):
                    print("create dir: ", path)
                    os.makedirs(path, exist_ok=False)
                
                f_name = Path(f).name
                mask.save("data/google/masks/{0}/{1}".format(str(i),f_name))


if __name__ == "__main__":
    main()
