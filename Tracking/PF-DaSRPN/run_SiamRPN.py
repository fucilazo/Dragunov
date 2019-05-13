# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import numpy as np
import heapq
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import get_subwindow_tracking


# 构造锚点数组
def generate_anchor(total_stride, scales, ratios, score_size):
    # scale为8，需要根据输入小心设计
    anchor_num = len(ratios) * len(scales)  # ratios = [0.33, 0.5, 1, 2, 3]     scales = [8, ]
    anchor = np.zeros((anchor_num, 4),  dtype=np.float32)
    # [[0., 0., 0., 0.],
    #  [0., 0., 0., 0.],
    #  [0., 0., 0., 0.],
    #  [0., 0., 0., 0.],
    #  [0., 0., 0., 0.]]

    # size似乎改成 Receptive Field 更好理解     即第n层特征图中一个像素，对应第1层（输入图像）的像素数
    size = total_stride * total_stride      # 8*8=64
    count = 0
    for ratio in ratios:
        # ws = int(np.sqrt(size * 1.0 / ratio))
        ws = int(np.sqrt(size / ratio))     # 13，11，8，5，4
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1

    # 对锚点组进行广播，并设置其坐标       tile() 函数,就是将原矩阵横向、纵向地复制
    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    # 加上ori偏移后，xx和yy以图像中心为原点
    ori = - (score_size / 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    # print(anchor.shape)   # (1805, 4)
    return anchor


# 跟踪器参数
class TrackerConfig(object):
    # These are the default hyper-params for DaSiamRPN 0.3827
    windowing = 'cosine'  # to penalize large displacements [cosine/uniform]
    # Params from the network architecture, have to be consistent with the training
    exemplar_size = 127  # input z size
    instance_size = 271  # input x size (search region)     # 目标占比小于0.4%时变为287
    total_stride = 8
    score_size = (instance_size-exemplar_size)/total_stride+1
    context_amount = 0.5  # context amount for the exemplar
    ratios = [0.33, 0.5, 1, 2, 3]
    scales = [8, ]
    anchor_num = len(ratios) * len(scales)
    anchor = []
    penalty_k = 0.055
    window_influence = 0.42
    lr = 0.295
    # adaptive change search region #
    adaptive = True     # 将根据目标和输入图像的大小调整搜索区域

    def update(self, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)
        self.score_size = (self.instance_size - self.exemplar_size) / self.total_stride + 1


# 检测分支
def tracker_eval(net, x_crop, target_pos, target_sz, window, scale_z, p, im, zoom):
    # 运行网络的检测分支，得到坐标回归量和得分
    # print(net(x_crop))
    delta, score = net(x_crop)

    # print(delta.shape)  # torch.Size([1, 20, 19, 19])
    # print(score.shape)  # torch.Size([1, 10, 19, 19])

    # 置换delta，其形状由 N x 4k x H x W 变为4x(kx17x17)。score形状为2x(kx17x17)，并取其后一半结果
    # torch.Tensor.permute 置换此张量的尺寸
    # torch.Tensor.contiguous 返回包含与自张量相同的数据的连续张量。如果自张量是连续的，则此函数返回自张量
    # torch.Tensor.numpy 将自张量作为 NumPy ndarray 返回。此张量和返回的 ndarray 共享相同的底层存储。自张量的变化将反映在 ndarray 中，反之亦然
    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy()

    # print(delta.shape)  # (4, 1805)
    # print(score.shape)  # (1805,)

    # 由于p.anchor[:, 0]和p.anchor[:, 1]采用相对坐标，所以delta[0, :]和delta[1, :]表示相对前一帧的中心偏移，而delta[2, :]和delta[3, :]为预测宽高
    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]     # X偏移
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]     # Y偏移
    delta[2, :] = (np.exp(delta[2, :]) * p.anchor[:, 2])            # W
    delta[3, :] = (np.exp(delta[3, :]) * p.anchor[:, 3])            # H

    # v-------------------------------------------------------------------------------------------------------------
    # 显示每一帧锚框
    # img = np.ones((360, 480, 3))  # 128*128的白色图片
    # plt.imshow(img)
    # Axs = plt.gca()
    # for i in range(delta.shape[1]):
    #     rec = patches.Rectangle((delta[0][i] + target_pos[0], delta[1][i] + target_pos[1]), 20, 20, edgecolor='r', facecolor='none')
    #     Axs.add_patch(rec)
    # plt.show()

    # 显示锚框中心点
    # for i in range(delta.shape[1]):
    #     cv2.circle(im, (int(delta[0][i] + target_pos[0]), int(delta[1][i] + target_pos[1])), 1, (0, 0, 255), 0)
    # ^-------------------------------------------------------------------------------------------------------------

    # 尺寸惩罚。sz 和 sz_wh 分别计算两种输入类型的等效边长
    def change(r):
        return np.maximum(r, 1./r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    # size penalty
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz)))  # scale penalty
    r_c = change((target_sz[0] / target_sz[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
    pscore = penalty * score

    # window float      pscore按一定权值叠加一个窗分布值。找出最优得分的索引
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    best_pscore_id = np.argmax(pscore)  # 输出最大值所对应索引
    # v-------------------------------------------------------------------------------------------------------------
    # print(heapq.nlargest(5, range(len(pscore)), pscore.__getitem__))
    # best_pscore_ids = heapq.nlargest(10, range(len(pscore)), pscore.__getitem__)
    # target_temp = []
    # for i in range(10):
    #     target_temp.append(delta[:, best_pscore_ids[i]])
    # print(np.sum(target_temp, axis=0))
    # ^-------------------------------------------------------------------------------------------------------------
    # 获得目标的坐标及尺寸。delta除以scale_z映射到原图
    # v-------------------------------------------------------------------------------------------------------------
    # target = (np.sum(target_temp, axis=0))/10
    # target = target / scale_z
    target = delta[:, best_pscore_id] / scale_z
    # ^-------------------------------------------------------------------------------------------------------------
    target_sz = target_sz / scale_z
    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr

    # 由预测坐标偏移得到目标中心，宽高进行滑动平均
    res_x = target[0] + target_pos[0]
    res_y = target[1] + target_pos[1]
    res_w = target_sz[0] * (1 - lr) + target[2] * lr
    res_h = target_sz[1] * (1 - lr) + target[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])
    return target_pos, target_sz, score[best_pscore_id]


# -------------------------
# im--->第一帧图像
# target_pos--->box中心坐标
# target_sz--->box大小
# -------------------------
def SiamRPN_init(im, target_pos, target_sz, net):
    state = dict()  # 创建一个字典
    p = TrackerConfig()     # 初始化Tracker对象
    p.update(net.cfg)       # 为不同的net（model）作更新
    state['im_h'] = im.shape[0]     # 整幅图像的 高
    state['im_w'] = im.shape[1]     # 整幅图像的 宽

    if p.adaptive:
        # 根据目标和输入图像的大小调整搜索区域
        if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:  # 目标面积占比小于 0.4%
            p.instance_size = 287  # small object big search region
        else:
            p.instance_size = 271

        p.score_size = (p.instance_size - p.exemplar_size) / p.total_stride + 1     # (271-127)/8 + 1 = 19

    # 构造出以图像中心为原点，格式为[cx, cy, w, h]的锚点矩阵
    p.anchor = generate_anchor(p.total_stride, p.scales, p.ratios, int(p.score_size))
    # for i in range(p.anchor.shape[0]):
    #     box = p.anchor[i]
    #     cv2.rectangle(im, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 255), 0)
    # cv2.imshow('im', im)
    # cv2.waitKey(0)

    avg_chans = np.mean(im, axis=(0, 1))

    # p.context_amount * sum(target_sz)为填充边界。wc_z和hc_z表示纹理填充后的宽高，s_z为等效边长。
    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))   # 202
    # initialize the exemplar   填充并截取出目标
    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

    # 包裹张量并记录应用于它的操作
    z = Variable(z_crop.unsqueeze(0))
    # 运行 temple 函数计算模板结果
    net.temple(z.cuda())

    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)

    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    # --------------------------------------------------------------------------------------------------------------
    state['s_z_original'] = s_z
    state['s_x'] = 0
    # --------------------------------------------------------------------------------------------------------------
    return state


# 返回跟踪框左上角坐标
def SiamRPN_track(state, im):
    # 从state中获取所需变量
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']
    # v-------------------------------------------------------------------------------------------------------------
    s_z_original = state['s_z_original']
    # ^-------------------------------------------------------------------------------------------------------------

    # 计算扩展后尺寸   context_amount = 0.5, exemplar_size = 127, instance_size = 271(287)
    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    # v-------------------------------------------------------------------------------------------------------------
    # 缩放系数
    zoom = s_z/s_z_original
    # ^-------------------------------------------------------------------------------------------------------------
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad     # 受 target_sz 大小影响

    # extract scaled crops for search region x at previous target position
    # 在前一个目标位置为搜索区域x提取缩放的截图
    x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))

    # tracker_eval 预测出新的位置和得分       .cuda()将内存中的数据复制到GPU显存中去
    # target_pos, target_sz, score = tracker_eval(net, x_crop.cuda(), target_pos, target_sz * scale_z, window, scale_z, p)
    target_pos, target_sz, score = tracker_eval(net, x_crop.cuda(), target_pos, target_sz * scale_z, window, scale_z, p, im, zoom)
    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos    # 返回跟踪框左上角坐标
    state['target_sz'] = target_sz
    state['score'] = score
    # v-------------------------------------------------------------------------------------------------------------
    state['s_x'] = round(s_x)
    # <-----形变时的模板更新----->
    print(score)
    # if score <= 0.8:
    #     z_crop = Variable(get_subwindow_tracking(im, target_pos, p.exemplar_size, round(s_x), avg_chans).unsqueeze(0))
    #     net.temple(z_crop.cuda())
    # ^-------------------------------------------------------------------------------------------------------------
    return state
