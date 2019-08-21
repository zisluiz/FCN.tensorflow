import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob
import TensorflowUtils as utils

DATA_URL = 'http://rgbd.cs.princeton.edu/data/SUNRGBD.zip'


def read_dataset(data_dir):
    pickle_filename = "splits.pkl"
    pickle_filepath = os.path.join(data_dir, pickle_filename)

    #sunrgbd_dir = "splits.pkl"
    #sunrgbd_dirpath = os.path.join(data_dir, sunrgbd_dir)

    #if not os.path.exists(sunrgbd_dirpath):
    #    utils.maybe_download_and_extract(data_dir, DATA_URL, is_zipfile=True)

    splits = pickle.load(open(pickle_filepath, "rb"), encoding="latin1")
    trainval = splits["trainval"]
    test = splits["test"]

    return create_image_lists(data_dir, 'train', trainval), create_image_lists(data_dir, 'test', test)

def create_image_lists(sunrgbd_dirpath, type, indexes):
    image_list = []

    for value in indexes:
        #print('index: '+ str(value))
        rgb_image_path = os.path.join(sunrgbd_dirpath, "images-224", str(value) + ".png")
        depth_image_path = os.path.join(sunrgbd_dirpath, "depth-inpaint-u8-224", str(value) + ".png")
        mask_path = os.path.join(sunrgbd_dirpath, "seglabel-224", str(value) + ".png")

        record = { 'image': rgb_image_path, 'annotation': mask_path, 'depth': depth_image_path, 'filename': str(value)+'.png' }
        #print(record)
        image_list.append(record)

    random.shuffle(image_list)
    no_of_images = len(image_list)
    print ('No. of %s files: %d' % (type, no_of_images))    
    return image_list

