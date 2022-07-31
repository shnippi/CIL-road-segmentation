"""
#
# This script removes all images that don't show enough road.
# We dod this by looking at the mask and if there is not enough road labeled pixels we remove the image from the dataset
#
"""


import os
from PIL import Image


def main():
    # open method used to open different extension image file
    k = 0
    for i in range(0,20):
        print("Starting directroy: ", i)
        directory = "data/google/masks/{0}".format(str(i))
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(f):
                
                mask = Image.open(f)
                num_white_pixels = _count_nonblack_pil(mask)
            
                # Delete if not enough pixels that are white
                if num_white_pixels < 200:
                    name = os.path.basename(f)

                    path_masks = f
                    path_roadmap = "data/google/roadmaps/{0}/".format(str(i)) + name
                    path_images = "data/google/images/{0}/".format(str(i)) + name

                    os.remove(path_masks)
                    os.remove(path_roadmap)
                    os.remove(path_images)
                

def _count_nonblack_pil(img):
    """Return the number of pixels in img that are not black.
    img must be a PIL.Image object in mode RGB.

    """
    bbox = img.getbbox()
    if not bbox: return 0
    return sum(img.crop(bbox)
               .point(lambda x: 255 if x else 0)
               .convert("L")
               .point(bool)
               .getdata())


if __name__ == "__main__":
    main()
