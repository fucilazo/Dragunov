import cv2
import numpy as np

import os
import sys
import tensorflow as tf

from PIL import Image
import scipy.misc

from utils import label_map_util
from utils import visualization_utils as vis_util


# --------------------------------------------------tensorflow--------------------------------------------------
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.chdir('C:\\Dragunov\\openCV\\extral\\Apply-Kalman_and_PF_in_blueballDemo')
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

MODEL_NAME = 'model'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph-50000.pb'
PATH_TO_LABELS = os.path.join('data', 'pascal_label_map.pbtxt')
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


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def detection(image):   # return Weight
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

            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # col the area
            y1 = np.squeeze(boxes)[0][0]
            x1 = np.squeeze(boxes)[0][1]
            y2 = np.squeeze(boxes)[0][2]
            x2 = np.squeeze(boxes)[0][3]
            area = (x2-x1)*(y2-y1)

            return np.squeeze(scores)[0] * area
# --------------------------------------------------tensorflow--------------------------------------------------


# 计算权重
def likelihood(x, y, func, image, w=100, h=100):
    # 以粒子为中心选取权重判定范围
    # 左上角坐标
    x1 = np.int32(max(0, x - w / 2))
    y1 = np.int32(max(0, y - h / 2))
    # 右上角坐标
    x2 = np.int32(min(image.shape[1], x + w / 2))
    y2 = np.int32(min(image.shape[0], y + h / 2))
    # cut the area
    region = image[y1:y2, x1:x2]    # 获取图像（30x30）-----根据粒子坐标
    # cv2.imshow("region", region)
    # count the pixel ------（随着选取范围增大，会因为判定数量增大而影响速度）
    count = region[func(region)].size   # return TRUE.size（偶尔会出现FALSE.size ？）的数量（0~900）
    return (float(count) / image.size) if count > 0 else 0.0001     # 返回百分比


# 计算权重
def likelihood_tensor(x, y, func, image, w=30, h=30):
    # 以粒子为中心选取权重判定范围
    # 左上角坐标
    x1 = np.int32(max(0, x - w / 2))
    y1 = np.int32(max(0, y - h / 2))
    # 右上角坐标
    x2 = np.int32(min(image.shape[1], x + w / 2))
    y2 = np.int32(min(image.shape[0], y + h / 2))
    # cut the area
    region = image[y1:y2, x1:x2]    # 获取图像（30x30）-----根据粒子坐标
    pl_image = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
    # cv2.imshow("region", region)
    # count the pixel ------（随着选取范围增大，会因为判定数量增大而影响速度）
    # count = region[func(region)].size   # return TRUE.size（偶尔会出现FALSE.size ？）的数量（0~900）
    # return (float(count) / image.size) if count > 0 else 0.0001     # 返回百分比
    count = detection(pl_image)
    if count > 0:
        return count
    else:
        return 0.0001


def init_particles(func, image):
    mask = image.copy()
    mask[func(mask) == False] = 0
    # print(mask)
    # cv2.imshow("mask", mask)
    # 轮廓检测
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) <= 0:
        return None

    max_contour = max(contours, key=cv2.contourArea)
    max_rect = np.array(cv2.boundingRect(max_contour))
    max_rect = max_rect[:2] + max_rect[2:] / 2
    weight = likelihood(max_rect[0], max_rect[1], func, image)
    particles = np.ndarray((3, 3), dtype=np.float32)       # 粒子数量
    particles[:] = [max_rect[0], max_rect[1], weight]

    return particles


# resample according to the particles weight
def resample(particles):
    tmp_particles = particles.copy()
    weights = particles[:, 2].cumsum()
    last_weight = weights[weights.shape[0] - 1]
    for i in range(particles.shape[0]):
        weight = np.random.rand() * last_weight
        particles[i] = tmp_particles[(weights > weight).argmax()]
        particles[i][2] = 1.0


def predict(particles, variance=13.0):
    particles[:, 0] += np.random.randn((particles.shape[0])) * variance
    particles[:, 1] += np.random.randn((particles.shape[0])) * variance


def weight(particles, func, image):
    for i in range(particles.shape[0]):     # particles.shape[0] = 500      particles.shape[1] = 3 (x, y, weight)
        # print(particles.shape[1])
        particles[i][2] = likelihood(particles[i][0], particles[i][1], func, image)     # 给每个粒子赋权值，存储到particles[:][2]
        # adding the weight
        sum_weight = particles[:, 2].sum()
        particles[:, 2] *= (particles.shape[0] / sum_weight)


def weight_tensor(particles, func, rgb_image):
    for i in range(particles.shape[0]):     # particles.shape[0] = 500      particles.shape[1] = 3 (x, y, weight)
        # print(particles.shape[1])
        particles[i][2] = likelihood_tensor(particles[i][0], particles[i][1], func, rgb_image)     # 给每个粒子赋权值，存储到particles[:][2]
        # adding the weight
        sum_weight = particles[:, 2].sum()
        particles[:, 2] *= (particles.shape[0] / sum_weight)


# 估计出追踪中心坐标(x, y)
def measure(particles):
    x = (particles[:, 0] * particles[:, 2]).sum()
    y = (particles[:, 1] * particles[:, 2]).sum()
    weight = particles[:, 2].sum()
    return x / weight, y / weight


# get the cords of particle
particle_filter_cur_frame = 0


def particle_filter(particles, func, image, rgb, max_frame=10):
    global particle_filter_cur_frame

    if image[func(image)].size <= 0:
        if particle_filter_cur_frame >= max_frame:
            return None, -1, -1
        particle_filter_cur_frame = min(particle_filter_cur_frame + 1, max_frame)
    else:
        particle_filter_cur_frame = 0
        # 初始化粒子
        if particles is None:
            particles = init_particles(func, image)

    if particles is None:
        return None, -1, -1

    resample(particles)
    predict(particles)
    weight_tensor(particles, func, rgb)
    x, y = measure(particles)
    return particles, x, y


if __name__ == '__main__':
    def is_color(region):
        # limit the blur color range
        return (region >= 90) & (region < 130)

    # input the vedio
    cap = cv2.VideoCapture('ball6.mp4')
    # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('bb.mp4')

    # 第一帧粒子设为空，在particle_filter方法中调用init_particles方法初始化粒子
    particles = None

    while cv2.waitKey(20) < 1:
        _, frame = cap.read()   # frame为读取到的每一帧图像
        # Transfer to HSV format
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_h = frame_hsv[:, :, 0]
        # 固定阈值二值化   <---调整阈值分离出目标图像--->
        # 灰度值小于0的置0，大于0的置255
        _, frame_s = cv2.threshold(frame_hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        _, frame_v = cv2.threshold(frame_hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # 取得目标二值化图像
        frame_h[(frame_s == 0) | (frame_v == 0)] = 0

        # 二值化图像进行粒子滤波
        # particles, x, y = particle_filter(particles, is_color, frame_h)
        particles, x, y = particle_filter(particles, is_color, frame_h, frame)

        if particles is not None:
            # 粒子坐标
            valid_particles = np.int32(particles[(particles[:, 0] >= 0) & (particles[:, 0] < frame.shape[1]) &
                                                 (particles[:, 1] >= 0) & (particles[:, 1] < frame.shape[0])])
            # modify the color of particles
            for i in range(valid_particles.shape[0]):
                frame[valid_particles[i][1], valid_particles[i][0]] = [0, 255, 255]     # 粒子颜色
            p = np.array([x, y], dtype=np.int32)    # 跟踪框中心点坐标

            cv2.rectangle(frame, tuple(p - 25), tuple(p + 25), (0, 0, 255), thickness=2)

        cv2.imshow('green', frame)
        # cv2.imshow('frame_hsv', frame_hsv)
        # cv2.imshow('frame_h', frame_h)

    cap.release()
    cv2.destroyAllWindows()
