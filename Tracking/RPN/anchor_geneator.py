import numpy as np
# np.set_printoptions(threshold=np.inf)     # 取消numpy数组打印的省略
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# feature map size
size_y = 16
size_x = 16
# anchor滑动步长。即原始图像和feature map的缩放比（feature map上移动一个像素点相当于原始图像上移动 x 个像素点）
rpn_stride = 8  # 原始图像的size：(16*8)*(16*8)
# anchor的尺寸
scales = [2, 4, 8]  # 改小可以使锚框更加分明
# 锚框的形状（比例）（长的、方的、扁的）
ratios = [0.5, 1, 2]


def anchor_generator(size_y, size_x, rpn_stride, scales, ratios):
    scales, ratios = np.meshgrid(scales, ratios)            # 排列组合
    scales, ratios = scales.flatten(), ratios.flatten()     # 放平，变成一维数组
    # anchor的尺寸
    scale_y = scales * np.sqrt(ratios)
    scale_x = scales / np.sqrt(ratios)
    # 原始图片上anchor的坐标
    shift_x = np.arange(0, size_x) * rpn_stride
    shift_y = np.arange(0, size_y) * rpn_stride
    # 组合成网格，生成锚点坐标
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # 每个锚点放置各种尺寸的锚框
    center_x, anchor_x = np.meshgrid(shift_x, scale_x)
    center_y, anchor_y = np.meshgrid(shift_y, scale_y)
    # print(center_x.shape)   # (9, 256)  256个锚点，每个锚点有9个锚框。共有 9*256=2304 个锚框

    # 组合坐标
    anchor_center = np.stack([center_y, center_x], axis=2).reshape(-1, 2)   # reshape：每个元素只包含两个坐标
    # 组合尺寸
    anchor_size = np.stack([anchor_y, anchor_x], axis=2).reshape(-1, 2)
    # 获得锚框左上角和右下角的坐标
    boxes = np.concatenate([anchor_center - 0.5*anchor_size, anchor_center + 0.5*anchor_size], axis=1)

    return boxes    # (坐上顶点坐标, 右下顶点坐标, 尺寸, 尺寸)


if __name__ == '__main__':
    img = np.ones((128, 128, 3))    # 128*128的白色图片
    plt.imshow(img)
    Axs = plt.gca()   # 获取plt接口
    # 生成anchor
    anchors = anchor_generator(size_y, size_x, rpn_stride, scales, ratios)

    # 遍历生成anchors
    for i in range(anchors.shape[0]):
        box = anchors[i]
        rec = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], edgecolor='r', facecolor='none')
        Axs.add_patch(rec)

    plt.show()
