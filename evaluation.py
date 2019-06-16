import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os

def _transform(filename, __channels):
    image_options = {'resize': True, 'resize_size': 224}
    image = misc.imread(filename, flatten=False if __channels else True, mode='RGB' if __channels else 'P')
    if __channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
        image = np.array([image for i in range(3)])

    if image_options.get("resize", False) and image_options["resize"]:
        resize_size = int(image_options["resize_size"])
        resize_image = misc.imresize(image,
                                        [resize_size, resize_size], interp='nearest')
        print('resized')
    else:
        resize_image = image

    #array = np.array(resize_image)
    #array[array[:,:,3]==255,:3]

    return resize_image.reshape(1, resize_size,resize_size,3 if __channels else 1)

if __name__ == '__main__':
    rgb_filename = '/opt/project/tests/dataset/png/rgb/rgb_00000.png'
    depth_filename = '/opt/project/tests/dataset/png/depth/depth_00000.png'

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], 'logs2/saved_model')
        graph = tf.get_default_graph()
        pred = graph.get_tensor_by_name("ExpandDims:0")
        output = sess.run(pred, #'pred_annotation:0'
                feed_dict={'input_image:0': _transform(rgb_filename, True), 'annotation:0': 
            np.expand_dims(_transform(depth_filename, False), axis=3), 'keep_probabilty:0': 1.0})
        output = np.squeeze(output, axis=3)
        misc.imsave(os.path.join('/opt/project/tests/results/', 'pred_00000.png'), output[0].astype(np.uint8))