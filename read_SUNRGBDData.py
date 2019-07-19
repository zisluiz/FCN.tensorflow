import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob
import TensorflowUtils as utils

DATA_URL = 'http://rgbd.cs.princeton.edu/data/SUNRGBD.zip'


def read_dataset(data_dir):
    pickle_filename = "splits.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)

    sunrgbd_dir = "splits.pickle"
    self.sunrgbd_dirpath = os.path.join(data_dir, sunrgbd_dir)

    if not os.path.exists(sunrgbd_dirpath):
        utils.maybe_download_and_extract(data_dir, DATA_URL, is_zipfile=True)

    splits = pickle.load(open(pickle_filepath, "rb"), encoding="latin1")
    self.trainval = splits["trainval"]
    self.test = splits["test"]

    return create_image_lists('train', self.trainval), create_image_lists('test', self.test)

def create_image_lists(type, indexes):
    image_list = []

    for key in indexes:
        value = indexes[key]
        print('index: '+ value)
        rgb_image_path = os.path.join(self.sunrgbd_dirpath, "images-224", str(value) + ".png")
        depth_image_path = os.path.join(self.sunrgbd_dirpath, "depth-inpaint-u8-224", str(value) + ".png")
        mask_path = os.path.join(self.sunrgbd_dirpath, "seglabel-224", str(value) + ".png")

        record = { 'image': rgb_image, 'annotation': mask, 'depth': depth_image, 'filename': str(value)+'.png' }
        #print(record)
        image_list.append(record)

    random.shuffle(image_list)
    no_of_images = len(image_list)
    print ('No. of %s files: %d' % (type, no_of_images))    
    return image_list

