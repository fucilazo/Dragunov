# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python

import glob
import sys
import cv2  # imread
import torch
import numpy as np
import os
from os.path import realpath, dirname, join

from net import SiamRPNBIG
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect

temp_name = 0

# load net
net_file = join(realpath(dirname(__file__)), 'SiamRPNBIG.model')
net = SiamRPNBIG()
# torch.load 从文件加载用 torch.save() 保存的对象。load_state_dict() 将state_dict中的参数和缓冲区复制到此模块及其后代中
net.load_state_dict(torch.load(net_file))
# eval() 设置模块为测试模式。cuda(device=None) 将所有模型参数和缓冲区移动到GPU
net.eval().cuda()

# warm up
# 预热。运行模板和检测分支10次
for i in range(10):
    net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda())
    net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())

# image_files = sorted(glob.glob('./bag/*.jpg'))
# image_files = sorted(glob.glob('./helicopter/*.jpg'))
# image_files = sorted(glob.glob('./ant/*.jpg'))
# image_files = sorted(glob.glob('./basketball/*.jpg'))
# image_files = sorted(glob.glob('./crossing/*.jpg'))
image_files = sorted(glob.glob('./CarScale/*.jpg'))
# start to track    来给第一帧的图像初始化坐标用的，在第一帧以后的后续帧，用来更新坐标信息
# init_rbox = [334.02,128.36,438.19,188.78,396.39,260.83,292.23,200.41]
# init_rbox = [345.86,216.9,576.78,222.71,572.99,373.17,342.07,367.35]
# init_rbox = [137.21,458.36,156.83,460.78,148.35,529.41,128.72,526.99]
# init_rbox = [423,169,423,273,463,169,463,273]
# init_rbox = [201,153,201,203,223,153,223,203]
init_rbox = [6,166,6,189,48,171,48,192]
[cx, cy, w, h] = get_axis_aligned_bbox(init_rbox)   # 第一帧起始box。将坐标数据转换成 RPN 的格式


if not image_files:
    sys.exit(0)

target_pos, target_sz = np.array([cx, cy]), np.array([w, h])    # box中心坐标和box高和宽
im = cv2.imread(image_files[0])  # HxWxC
# SiamRPN_init 构造状态结构体并运行模板分支
state = SiamRPN_init(im, target_pos, target_sz, net)  # init tracker

for f, image_file in enumerate(image_files):
    if not image_file:
        break
    im = cv2.imread(image_file)  # HxWxC
    # 运行检测分支并更新状态变量
    state = SiamRPN_track(state, im)  # track
    # 将坐标转换成矩形框的表示形式
    res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
    # v-------------------------------------------------------------------------------------------------------------
    s_x = state['s_x']
    # print(s_x)
    # ^-------------------------------------------------------------------------------------------------------------
    res = [int(l) for l in res]
    # (res[0], res[1])为左上角坐标，res[2]、res[3]为宽和高
    cv2.rectangle(im, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 3)
    # v-------------------------------------------------------------------------------------------------------------
    center_pos = (res[0] + int(res[2]/2), res[1] + int(res[3]/2))
    # Bbox中心点
    cv2.circle(im, center_pos, 1, (0, 0, 0), 4)
    # 带背景的截取区域
    cv2.rectangle(im, (center_pos[0] - int(s_x/2), center_pos[1] - int(s_x/2)), (center_pos[0] + int(s_x/2), center_pos[1] + int(s_x/2)), (255, 0, 0), 1)
    # ^-------------------------------------------------------------------------------------------------------------
    cv2.imshow('SiamRPN', im)
    # v-------------------------------------------------------------------------------------------------------------
    # 按 S 键保存图片
    if cv2.waitKey(1) & 0xFF == ord('s'):
        print('Saving the image...')
        cv2.imwrite(str(temp_name) + '.jpg', im)
        print('Saving done.')
        temp_name = temp_name + 1
    # ^-------------------------------------------------------------------------------------------------------------
    # cv2.waitKey(1)

