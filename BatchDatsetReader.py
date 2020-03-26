"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc
import cv2
import random

class BatchDatset:
    files = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, image_options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list
        self.image_options = image_options
        self.total_files = len(self.files)

    def load_images(self, indexes, indexesCrop):
        images = []
        for i, index in enumerate(indexes):
            image = cv2.imread(self.files[index]['image'], cv2.IMREAD_COLOR)
            #image = misc.imread(self.files[index]['image'])
            image = self.crop(image, indexesCrop[i])
            image = self._transform_image(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        return images

    def load_annotations(self, indexes, indexesCrop):
        images = []
        for i, index in enumerate(indexes):
            image = cv2.imread(self.files[index]['annotation'], cv2.IMREAD_GRAYSCALE)
            #image = misc.imread(self.files[index]['annotation'])
            image = self.crop(image, indexesCrop[i])
            image = self._transform_image(image)
            images.append(np.expand_dims(image, axis=2))
        return images

    def crop(self, image, cropIndex):
        h, w = image.shape[:2]

        less_value = h if h < w else w

        if less_value > 448:
            less_value = 448
        else:
            less_value = 336

        if h > (less_value*2) or w > (less_value*2):
            raise Exception('Problem')

        if cropIndex == 0:
            image = image[0:less_value, 0:less_value]
        elif cropIndex == 1:
            image = image[0:less_value, w-less_value:w]
        elif cropIndex == 2:
            image = image[h-less_value:h, 0:less_value]
        elif cropIndex == 3:
            image = image[h-less_value:h, w-less_value:w]

        if image.shape[:2] != (less_value, less_value):
            raise Exception('Problem')

        return image

    def _transform(self, filename):
        image = misc.imread(filename)
        if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image,
                                         [resize_size, resize_size], interp='nearest')
        else:
            resize_image = image

        return np.array(resize_image)

    def _transform_image(self, image):
        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = cv2.resize(image, (resize_size, resize_size), interpolation=cv2.INTER_NEAREST)
            #resize_image = misc.imresize(image,
            #                             [resize_size, resize_size], interp='nearest')
        else:
            resize_image = image

        return resize_image

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.total_files:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            #perm = np.arange(self.images.shape[0])
            #np.random.shuffle(perm)
            #self.images = self.images[perm]
            #self.annotations = self.annotations[perm]
            random.shuffle(self.files)
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        indexes = list(range(start, end))
        cropes = np.random.randint(0, 4, size=[batch_size]).tolist()
        return self.load_images(indexes, cropes), self.load_annotations(indexes, cropes)

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.total_files, size=[batch_size]).tolist()
        indexesCrop = np.random.randint(0, 4, size=[batch_size]).tolist()
        return self.load_images(indexes, indexesCrop), self.load_annotations(indexes, indexesCrop)