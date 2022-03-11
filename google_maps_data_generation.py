#!/usr/bin/env python
# City location and code adaption taken from https://github.com/Walon1998/RoadsegmentSubmission
import glob
import io
import os
import numpy as np
import requests
import sys
from random import uniform
from PIL import Image
from natsort import natsorted
from skimage.io import imsave

NUM_SAMPLES = 1000

los_angeles = [
    {'n': 34.269260, 'w': -118.604202, 's': 34.171040, 'e': -118.370722},
    {'n': 34.100406, 'w': -118.362530, 's': 33.797995, 'e': -117.863483},
    {'n': 33.714559, 'w': -118.033473, 's': 33.636157, 'e': -117.746060}
]

chicago = [
    {'n': 42.072123, 'w': -88.311501, 's': 41.643560, 'e': -87.682533}
]

houston = [
    {'n': 29.875249, 'w': -95.563377, 's': 29.610542, 'e': -95.189842}
]

phoenix = [
    {'n': 33.688554, 'w': -112.381892, 's': 33.392095, 'e': -111.887507}
]

philadelphia = [
    {'n': 40.052889, 'w': -75.233393, 's': 39.904511, 'e': -75.140009},
    {'n': 40.049736, 'w': -75.144129, 's': 40.026079, 'e': -75.027399}
]

san_francisco = [
    {'n': 37.801910, 'w': -122.506267, 's': 37.737590, 'e': -122.398120},
    {'n': 37.826862, 'w': -122.295123, 's': 37.800282, 'e': -122.255984}
]

boston = [
    {'n': 42.387338, 'w': -71.141267, 's': 42.283792, 'e': -71.046510}
]

# The cities from which we choose images
cities_boxes = [los_angeles, chicago, houston, phoenix, philadelphia, san_francisco, boston]

def get_api_key():
    with open("api_key.txt", "r") as file:
        first_line = file.readline()
        print(first_line)
        return first_line


API_KEY = get_api_key()

# Pick a random 400x400 patch from one of the cities
def pick_random_image_from_city(x, y, zoom_=18, width_=400, height_=400):
    url = "https://maps.googleapis.com/maps/api/staticmap?"
    center = "center=" + str(x) + "," + str(y)
    zoom = "&zoom=" + str(zoom_)
    size = "&size=" + str(width_) + "x" + str(height_)
    sat_maptype = "&maptype=satellite"
    road_maptype = "&maptype=roadmap"
    no_banners = "&style=feature:all|element:labels|visibility:off"
    api_key = "&key=" + API_KEY

    sat_url = url + center + zoom + size + sat_maptype + no_banners + api_key
    road_url = url + center + zoom + size + road_maptype + no_banners + api_key

    sat_tmp = Image.open(io.BytesIO(requests.get(sat_url).content))
    road_tmp = Image.open(io.BytesIO(requests.get(road_url).content))
    sat_image = np.array(sat_tmp.convert('RGB'))
    roadmap = np.array(road_tmp.convert('RGB'))

    return sat_image, roadmap

# Save images and mask in the appropriate location
def save_image(image, roadmap):
    img_path = "./data/google/images/{}.png".format(counter)
    roadmap_path = "./data/google/roadmaps/{}.png".format(counter)
    imsave(img_path, image)
    imsave(roadmap_path, roadmap)


# Find out how many images there already are such that we dont overwrite any.
image_list = glob.glob('./data/google/images/*')
if len(image_list) == 0:
    counter = 0
else:
    last_image =  natsorted(image_list)[-1]
    counter = int(os.path.basename(last_image).split(".")[0]) + 1


# Safety measure for too many samples
if NUM_SAMPLES > 1000:
    sys.exit("You request > 1000 pictures!. We exit the program. Either comment this line or request less samples")


# Propose NUM_SAMPLES newly generated images
print("Starting API requests:")
for i in range(NUM_SAMPLES):
    city_nr = np.random.randint(len(cities_boxes))  # pick a city
    index = np.random.randint(len(cities_boxes[city_nr]))
    box = cities_boxes[city_nr][index]

    rand_x = uniform(box['w'], box['e'])
    rand_y = uniform(box['n'], box['s'])

    image, roadmap = pick_random_image_from_city(rand_y, rand_x)

    save_image(image, roadmap)

    counter = counter + 1
