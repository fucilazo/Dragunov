import time

import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import tensorflow as tf

from PIL import Image
import scipy.misc

# if tf.__version__ < '1.4.0':
#     raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

os.chdir('C:\\Dragunov\\tensorflow\\object_detection_API\\文件准备')

# Env setup
# This is needed to display the images.
# %matplotlib inline

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Object detection imports
from utils import label_map_util

from utils import visualization_utils as vis_util

# 这是我们刚才训练的模型,修改为自己输出的模型目录名
MODEL_NAME = 'model'

# 对应的Frozen model位置
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph-50000.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'pascal_label_map.pbtxt')

# 改成自己例子中的类别数，1
NUM_CLASSES = 1

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


# Detection

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# 测试图片位置
PATH_TO_TEST_IMAGES_DIR = os.getcwd() + '/test_images'
os.chdir(PATH_TO_TEST_IMAGES_DIR)
TEST_IMAGE_PATHS = os.listdir(PATH_TO_TEST_IMAGES_DIR)

# Size, in inches, of the output images.
# IMAGE_SIZE = (12, 8)
# 修改为自己的目录，必须存在该文件夹
output_path = 'C:\\Dragunov\\tensorflow\\object_detection_API\\文件准备\\'

start = time.time()
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        for image_path in TEST_IMAGE_PATHS:
            image = Image.open(image_path)
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # print(image_np)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),  # [y1, x1, y2, x2]
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            # 保存文件
            im = scipy.misc.toimage(image_np, cmin=0.0, cmax=...)
            im.save(output_path + image_path.split('\\')[-1])

            # --------------------TEST--------------------
            # print(np.squeeze(scores)[0])
            # print(np.squeeze(boxes)[0])
            # y1 = np.squeeze(boxes)[0][0]
            # x1 = np.squeeze(boxes)[0][1]
            # y2 = np.squeeze(boxes)[0][2]
            # x2 = np.squeeze(boxes)[0][3]


end = time.time()
print("Execution Time: ", end - start)

