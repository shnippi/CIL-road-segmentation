from PIL import Image
import glob
import io
import os
import matplotlib.pyplot as plt
from skimage.io import imsave
import numpy as np
from tqdm import tqdm
from pathlib import Path

## ------------------------------------------------------------------ PARAMS START -----------------------------------------------------------

# number of samples taken for the avg reference image
NUMBER_OF_IMAGES_TARGET = 70
NUMBER_OF_IMAGES_TO_NORMALIZE = 1

# only takes left image if it's a double image
IS_DOUBLE_IMAGE = True

# for debug, shows matplotlib plots of histograms, cumsums and images
DRAW = False

# Path to images we want to match, reference images
PATH_TARGET = "data/cil-road-segmentation-2022/reference-images/images/" 

# path to images we want to normalize
PATH_TO_NORMALIZE = "data/google/images/test/"

# path to save normalised images
PATH_TO_SAVE = "data/google/images_normalized/test/"

## ------------------------------------------------------------------ PARAMS END -----------------------------------------------------------

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

N_BINS = 64
BIN_RES = 256/N_BINS

def create_histogram(flattened_image_channel):
    histogram = np.zeros((N_BINS))

    for pixel in flattened_image_channel:
        pixels_to_bin = np.floor(pixel/BIN_RES).astype(int)    
        histogram[pixels_to_bin] += 1/len(flattened_image_channel)
    return histogram

def ecdf(x):
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
    return vals, ecdf

## ------------------------------------------------------------------ UTIL FUNCTIONS END -----------------------------------------------------------


## random images
#IMAGES_TARGET = np.random.choice(sorted(create_path_list(PATH_TARGET)), size=(NUMBER_OF_IMAGES_TARGET))
#IMAGES_TO_NORMALIZE = np.random.choice(sorted(create_path_list(PATH_TO_NORMALIZE)), size=(NUMBER_OF_IMAGES_TO_NORMALIZE))


## all images
IMAGES_TARGET = sorted(create_path_list(PATH_TARGET))
IMAGES_TO_NORMALIZE = sorted(create_path_list(PATH_TO_NORMALIZE))


## load and average all target images
N_IMAGES_TARGET = len(IMAGES_TARGET)
AVG_IMAGE_TARGET = []
SHAPE_IMAGES_TARGET = (0, 0, 0)
ALL_IMAGES_REFERENCE = []

T_VALUES = np.array([i for i in range(256)])
T_COUNTS = np.zeros((3, 256))

for indx, image_path in tqdm(enumerate(IMAGES_TARGET)):
    # load image
    image = np.asarray(Image.open(image_path).convert('RGB'))
    
    if indx == 0:
        SHAPE_IMAGES_TARGET = image.shape
        if IS_DOUBLE_IMAGE:
            SHAPE_IMAGES_TARGET = (SHAPE_IMAGES_TARGET[0], SHAPE_IMAGES_TARGET[1]//2, SHAPE_IMAGES_TARGET[2])
        AVG_IMAGE_TARGET = np.zeros(SHAPE_IMAGES_TARGET)
        print(f"TARGET IMAGE SIZE = {SHAPE_IMAGES_TARGET}")

        # drop right side of image
    if(IS_DOUBLE_IMAGE):
        image = image[:,:SHAPE_IMAGES_TARGET[1],:]

    for channel in range(3):
        t_value, t_counts = np.unique(np.ravel(image[:,:,channel]), return_counts=True)
        for i, _ in enumerate(t_value):
            T_COUNTS[channel, t_value[i]] += t_counts[i]

    ALL_IMAGES_REFERENCE.append(image)

    if(indx == 0):
        AVG_IMAGE_TARGET = AVG_IMAGE_TARGET + image


ALL_IMAGES_REFERENCE = np.array(ALL_IMAGES_REFERENCE)
AVG_IMAGE_TARGET = (AVG_IMAGE_TARGET).astype(np.uint8)

if DRAW:
    reference_histogram = []
    x1 = []
    y1 = []

    for channel in range(3):
        reference_channel = np.ravel(ALL_IMAGES_REFERENCE[:,:,:,channel])
        reference_histogram.append(create_histogram(reference_channel))
        x0, y0 = ecdf(reference_channel[i])
        x1.append(x0)
        y1.append(y0)

## loop over all images that need to be normalized
for indx, image_path in tqdm(enumerate(IMAGES_TO_NORMALIZE)):
    # load image
    image = np.asarray(Image.open(image_path).convert('RGB'))
    image_shape = image.shape

    new_image = np.zeros(image_shape).astype(np.uint8)

    for channel in range(3):
        channel_shape = (image[:,:,channel]).shape
        flattened_image_channel = np.ravel(image[:,:,channel])

        # do histograms of unique pixels
        s_values, bin_idx, s_counts = np.unique(flattened_image_channel, return_inverse=True,
                                            return_counts=True)

        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(T_COUNTS[channel]).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        interp_t_values = np.interp(s_quantiles, t_quantiles, T_VALUES)

        new_image_channel = (interp_t_values[bin_idx].reshape(channel_shape)).astype(np.uint8)
        new_image[:,:,channel] = new_image_channel

    #save new image
    im = Image.fromarray(new_image)
    new_path = image_path.split('/')
    new_path[-4] = 'images_normalized'
    save_path = "/".join(new_path)
    Path("/".join(save_path.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
    im.save(save_path)

    if DRAW:
        fig, axs = plt.subplots(3, 3, figsize=(20,15))
        fig.suptitle(f"Image: {image_path}")

        # draw images
        axs[0][0].imshow(AVG_IMAGE_TARGET)
        axs[1][0].title.set_text("reference image")

        axs[0][1].imshow(image)
        axs[1][1].title.set_text("original image")

        axs[0][2].imshow(new_image)
        axs[1][2].title.set_text("normalized image")

        # draw histograms & cumsum
        BIN_VALUES = [i*256/N_BINS for i in range(N_BINS)]
        for i, band in enumerate(['red', 'green', 'blue']):
            
            original_channel = np.ravel(np.ravel(image[:,:,i]))
            normalised_channel = np.ravel(np.ravel(new_image[:,:,i]))

            
            original_histogram = create_histogram(original_channel)
            normalised_histogram = create_histogram(normalised_channel)

            axs[1][i].hist(BIN_VALUES, weights=reference_histogram[i], bins=N_BINS, alpha=0.5, label='reference')
            axs[1][i].hist(BIN_VALUES, weights=original_histogram, bins=N_BINS, alpha=0.5, label='original')
            axs[1][i].hist(BIN_VALUES, weights=normalised_histogram, bins=N_BINS, alpha=0.5, label='normalised')
            axs[1][i].title.set_text(band)
            axs[1][i].legend()

           
            x2, y2 = ecdf(original_channel)
            x3, y3 = ecdf(normalised_channel)
        
            axs[2][i].plot(x1[i], y1[i] * 100, '-r', lw=3, label='reference')
            axs[2][i].plot(x2, y2 * 100, '-b', lw=3, label='original')
            axs[2][i].plot(x3, y3 * 100, '--g', lw=3, label='normalised')

        plt.show()