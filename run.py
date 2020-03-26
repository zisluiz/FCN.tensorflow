import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os
import glob
import os.path as osp
from pathlib import Path
import psutil
from datetime import datetime
import time
import nvidia_smi
import cv2


def _transform(filename, __channels, datasetName):
    image_options = {'resize': True, 'resize_size': 224}
    #image = misc.imread(filename, flatten=False if __channels else True, mode='RGB' if __channels else 'P')
    image = cv2.imread(filename, cv2.IMREAD_COLOR if __channels else cv2.IMREAD_GRAYSCALE)

    if __channels:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if datasetName == "active_vision" or datasetName == "putkk":
        image = image[0:1080, 420:1500]
    elif datasetName == "semantics3d_raw":
        image = image[0:1024, 128:1152]

    if __channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
        image = np.array([image for i in range(3)])

    if image_options.get("resize", False) and image_options["resize"]:
        resize_size = int(image_options["resize_size"])
        #resize_image = misc.imresize(image,
        #                                [resize_size, resize_size], interp='nearest')
        resize_image = cv2.resize(image, (resize_size, resize_size), interpolation=cv2.INTER_NEAREST)
        #print('resized')
    else:
        resize_image = image

    #array = np.array(resize_image)
    #array[array[:,:,3]==255,:3]

    return resize_image.reshape(1, resize_size, resize_size, 3 if __channels else 1)

if __name__ == '__main__':
    useGpu = False

    fileName = "run_"
    if not useGpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        fileName += "cpu_"

    os.makedirs('results', exist_ok=True)
    f = open("results/"+fileName+str(int(round(time.time() * 1000)))+".txt", "w+")
    f.write('=== Start time: '+str(datetime.now())+'\n')

    p = psutil.Process(os.getpid())
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    graph = tf.Graph()
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], 'logs/saved_model/230000')
        graph = tf.get_default_graph()

        pred = graph.get_tensor_by_name("ExpandDims:0")

        print('Starting list image files')
        filesCount = 0

        files = glob.glob("datasets/mestrado/**/rgb/*.png", recursive=True)
        files.extend(glob.glob("datasets/mestrado/**/rgb/*.jpg", recursive=True))
        cpuTimes = [0.0, 0.0, 0.0, 0.0]

        gpuTimes = 0.0
        gpuMemTimes = 0.0
        maxNumThreads = 0
        memUsageTimes = 0

        for imagePath in files:
            print('imagePath: ' + imagePath)
            pathRgb = Path(imagePath)
            datasetName = osp.basename(str(pathRgb.parent.parent))
            # print('datasetName: ' + datasetName)
            parentDatasetDir = str(pathRgb.parent.parent)
            # print('parentDatasetDir: ' + parentDatasetDir)
            depthImageName = os.path.basename(imagePath).replace('jpg', 'png')

            output = sess.run(pred,  # 'pred_annotation:0'
                              feed_dict={'input_image:0': _transform(imagePath, True, datasetName), 'keep_probabilty:0': 1.0})
            output = np.squeeze(output, axis=3)

            res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            curGpuTime = res.gpu
            # curGpuMemTime = res.memory #(in percent)
            curGpuMemTime = mem_res.used / 1e+6
            gpuTimes += curGpuTime
            gpuMemTimes += curGpuMemTime
            f.write('GPU Usage Percent: ' + str(curGpuTime) + '\n')
            f.write('GPU Mem Usage (MB)): ' + str(curGpuMemTime) + '\n')

            curProcessCpuPerU = p.cpu_percent()
            curCpusPerU = psutil.cpu_percent(interval=None, percpu=True)

            # gives a single float value
            for i in range(len(cpuTimes)):
                curProcessCpu = curProcessCpuPerU
                curCpu = curCpusPerU[i]
                cpuTimes[i] += curCpu
                f.write('Process CPU Percent: ' + str(curProcessCpu) + ' --- CPU Percent: ' + str(curCpu) + '\n')

            # you can convert that object to a dictionary
            memInfo = dict(p.memory_full_info()._asdict())
            curMemUsage = memInfo['uss']
            memUsageTimes += curMemUsage

            f.write('Process memory usage: ' + str(curMemUsage / 1e+6) + '\n')
            f.write('Memory information: ' + str(memInfo) + '\n')

            if maxNumThreads < p.num_threads():
                maxNumThreads = p.num_threads()

            # print('############## Index: ')
            # print(index)
            os.makedirs('results/' + datasetName, exist_ok=True)
            misc.imsave('results/' + datasetName + '/' + depthImageName, output[0].astype(np.uint8))
            filesCount = filesCount + 1
        nvidia_smi.nvmlShutdown()

        start = time.time()
        for imagePath in files:
            pathRgb = Path(imagePath)
            datasetName = osp.basename(str(pathRgb.parent.parent))
            parentDatasetDir = str(pathRgb.parent.parent)
            depthImageName = os.path.basename(imagePath).replace('jpg', 'png')

            output = sess.run(pred,  # 'pred_annotation:0'
                              feed_dict={'input_image:0': _transform(imagePath, True, datasetName), 'keep_probabilty:0': 1.0})
            output = np.squeeze(output, axis=3)
        end = time.time()

        f.write('=== Mean GPU Usage Percent: ' + str(gpuTimes / filesCount) + '\n')
        f.write('=== Mean GPU Mem Usage (MB): ' + str(gpuMemTimes / filesCount) + '\n')
        for i in range(len(cpuTimes)):
            f.write("=== Mean cpu" + str(i) + " usage: " + str(cpuTimes[i] / filesCount) + '\n')
        f.write("=== Mean memory usage (MB): " + str((memUsageTimes / filesCount) / 1e+6) + '\n')

        f.write("=== Total image predicted: " + str(filesCount) + '\n')
        f.write("=== Seconds per image: " + str(((end - start) / filesCount)) + '\n')
        f.write("=== Max num threads: " + str(maxNumThreads) + '\n')

        f.write('=== End time: ' + str(datetime.now()) + '\n')
        f.close()

