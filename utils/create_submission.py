import os
import numpy as np
import re
import PIL
from PIL import Image
import argparse
import yaml
from skimage.color import rgb2gray
from pathlib import Path    

# assign a label to a patch
def _patch_to_label(patch):
    foreground_threshold = 0.25 # percentage of pixels of val 255 required to assign a foreground label to a patch
    patch = patch.astype(np.float64) / 255
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def _mask_to_submission_strings(image_filename, mask_dir=None):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    print(img_number)
    im = PIL.Image.open(image_filename)
    im_arr = np.asarray(im)
    if len(im_arr.shape) > 2:
        # Convert to grayscale.
        im = im.convert("L")
        im_arr = np.asarray(im)

    patch_size = 16
    mask = np.zeros_like(im_arr)
    for j in range(0, im_arr.shape[1], patch_size):
        for i in range(0, im_arr.shape[0], patch_size):
            patch = im_arr[i:i + patch_size, j:j + patch_size]
            label = _patch_to_label(patch)
            mask[i:i+patch_size, j:j+patch_size] = int(label*255)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))

    if mask_dir:
        _save_mask_as_img(mask, os.path.join(mask_dir, "mask_" + image_filename.split("/")[-1]))
    

def _save_mask_as_img(img_arr, mask_filename):
    img = PIL.Image.fromarray(img_arr)
    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
    img.save(mask_filename)


def _masks_to_submission(submission_filename, mask_dir, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in _mask_to_submission_strings(fn, mask_dir=mask_dir))


def _create_masks_from_roadmaps(config):
    print("Generating masks from roadmap")
    base_dir = config['result_dir'] + "/" + config['model'] + "_" + config['generation_mode'] + "_" + config['transformation']
    read_dir = base_dir + "/roadmaps"
    save_dir = base_dir + "/masks/"
    for filename in os.listdir(read_dir):
        f = os.path.join(read_dir, filename)
        # checking if it is a file
        if os.path.isfile(f):

            roadmap = Image.open(f) 
            roadmap = np.array(roadmap.convert('RGB'))
            f_thresh = np.vectorize(lambda x: 255 if x > 249/256 else 0)
            mask = rgb2gray(roadmap)
            mask = f_thresh(mask).astype(np.uint8)
            mask = np.expand_dims(mask, axis=2)
            mask = np.repeat(mask, 3, axis=2)
            mask = Image.fromarray(mask)

            f_name = Path(f).name
            mask.save(save_dir + f_name)


def create_submission(config):
    base_path = config['result_dir'] + "/" + config['model'] + "_" + config['generation_mode'] + "_" + config['transformation'] + "/masks"
    filename_path = base_path  + config['model'] + "_" + config['generation_mode'] + config['transformation'] + "_submisson.csv"

    if config['transformation'] == 'rgb':
        _create_masks_from_roadmaps(config)

    image_filenames = [os.path.join(base_path, name) for name in os.listdir(base_path)]
    _masks_to_submission(filename_path, "", *image_filenames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="configs/classic/test.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, "r"))
    create_submission(config)
