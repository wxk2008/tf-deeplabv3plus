#!--*-- coding:utf-8 --*--

# Deeplab Demo

import os
import numpy as np
from PIL import Image
import tempfile
from six.moves import urllib

import tensorflow as tf
import cv2
import time

class DeepLabModel(object):
    """
    加载 DeepLab 模型；
    推断 Inference.
    """
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """
        Creates deeplab model.
        """
        self.graph = tf.Graph()

        #graph_def = None

        graph_def = tf.GraphDef.FromString(open(tarball_path, 'rb').read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)


    def run(self, image):
        """
        Runs inference on a single image.

        Args:
        image: A PIL.Image object, raw input image.

        Returns:
        resized_image: RGB image resized from original input image.
        seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME,
                                      feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


def create_pascal_label_colormap():
    """
    Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
        A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """
    Adds color defined by the dataset colormap to the label.

    Args:
        label: A 2D array with integer type, storing the segmentation label.

    Returns:
        result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]

def write_result(imagefile, imgname):

    orignal_im = Image.open(imagefile)
    print('running deeplab on image %s...' % imagefile)
    resized_im, seg_map = MODEL.run(orignal_im)
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    img_path = "/home/ke/project/deeplab/output/" + "mask__" +  imgname
    cv2.imwrite(img_path, seg_image)


if __name__ == "__main__":
   LABEL_NAMES = np.asarray(['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                             'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                             'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv' ])
   FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
   FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
   model_dir = '/home/ke/project/deeplab/datasets/pascal_voc_seg/deeplabv3_mnv2_pascal_trainval/model.pb'
   MODEL = DeepLabModel(model_dir)
   print('model loaded successfully!')
   images_dir = '/home/ke/project/deeplab/images'
   images = sorted(os.listdir(images_dir))
   for imgfile in images:
     from_time = time.time()
     write_result(os.path.join(images_dir, imgfile), imgfile)
     to_time = time.time()
     decay_time = to_time - from_time
     print("the time of %s is %f"%(imgfile,decay_time))
   print('Done.')
