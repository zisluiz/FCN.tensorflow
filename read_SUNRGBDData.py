import numpy as np
import os
import random
import h5py
import cv2
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob
import TensorflowUtils as utils
import scipy.io

def read_dataset(data_dir):
    allsplit_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat')
    images_dir = os.path.join(data_dir, 'SUNRGBD')
    SUNRGBDMeta_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat')
    SUNRGBD2Dseg_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat')
    SUNRGBD2Dseg = h5py.File(SUNRGBD2Dseg_dir, mode='r', libver='latest')

    SUNRGBDMeta = scipy.io.loadmat(SUNRGBDMeta_dir, squeeze_me=True,
                                   struct_as_record=False)['SUNRGBDMeta']
    split = scipy.io.loadmat(allsplit_dir, squeeze_me=True, struct_as_record=False)
    trainval = split['alltrain']
    test = split['alltest']

    return create_image_lists(data_dir, SUNRGBDMeta, SUNRGBD2Dseg, trainval, test)

def create_image_lists(data_dir, SUNRGBDMeta, SUNRGBD2Dseg, split_train, split_test):
    image_list_train = []
    image_list_test = []

    seglabel = SUNRGBD2Dseg['SUNRGBD2Dseg']['seglabel']

    for i, meta in enumerate(SUNRGBDMeta):
        meta_dir = '/'.join(meta.rgbpath.split('/')[:-2])
        real_dir = meta_dir.replace('/n/fs/sun3d/data', data_dir)
        rgb_path = os.path.join(real_dir, 'image/' + meta.rgbname)

        label_path = os.path.join(real_dir, 'label/label.npy')
        label_img_path = os.path.join(real_dir, 'label/label.png')

        if not os.path.exists(label_img_path):
            os.makedirs(os.path.join(real_dir, 'label'), exist_ok=True)
            label = np.array(SUNRGBD2Dseg[seglabel.value[i][0]].value.transpose(1, 0))
            #np.save(label_path, label)
            cv2.imwrite(label_img_path, label)

        record = {'image': rgb_path, 'annotation': label_img_path, 'filename': str(meta.rgbname)}

        if meta_dir in split_train:
            image_list_train.append(record)
        else:
            image_list_test.append(record)

    random.shuffle(image_list_train)
    random.shuffle(image_list_test)
    no_of_images_train = len(image_list_train)
    no_of_images_test = len(image_list_test)
    print ('No. of %s train files: %d' % (type, no_of_images_train))
    print('No. of %s test files: %d' % (type, no_of_images_test))
    return image_list_train, image_list_test