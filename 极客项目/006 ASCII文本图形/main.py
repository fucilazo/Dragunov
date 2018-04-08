import sys, random, argparse
import numpy as np
import math
from PIL import Image

# grayscale level values from: http://paulbourke.net/dataformats/asciiart/
# 70 levels of gray
gscale1 = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
# 10 levels of gray
gscale2 = "@%#*+=-:. "


def getAverage(image):
    """
    :param image: Given PIL Image
    :return: average value of grayscale value
    """
    # get image as numpy array
    im = np.array(image)
    # 获取尺寸
    w, h = im.shape
    # get the average
    return np.average(im.reshape(w*h))


def covertImageToAscii(filename, cols, scale, moreLevels):
    """
    Given image and dimentions
    :param filename:
    :param cols:
    :param scale:
    :param moreLevels:
    :return: an m*n list of image
    """
    # 声明全局变量
    global gscale1, gscale2
    # 打开图像 并 转换为灰度图
    image = Image.open(filename).convert('L')
    # 获取图像尺寸
    W, H = image.size[0], image.size[1]
    print("input image dims: %dx%d" % (W, H))
    # compute title width
    w = W/cols
    # compute title height based on the aspect ratio and scale of the font
    h = w/scale
    # compute number of rows to use in the final grid
    rows = int(H/h)

    print("cols: %d, rows: %d" % (cols, rows))
    print("tile dims: %dx%d" % (W, H))

    # 检测图像是否太小
    if cols > W or rows > H:
        print("Image is too small for specified cols!")
        exit(0)

    # 初始化ASCII图像字符串列表
    aimg = []
    # generate the list of tile dimentions
    for j in range(rows):
        y1 = int(j*h)
        y2 = int((j+1)*h)
        # correct the last tile
        if j == rows-1:
            y2 = H
        # append an empty string
        aimg.append('')
        for i in range(cols):
            # crop the image to fit the tile
            x1 = int(i*w)
            x2 = int((i+1)*w)
            # correct the last tile
            if i == cols-1:
                x2 = W
            # crop the image to extract the tile into another Image object
            img = image.crop((x1, y1, x2, y2))
            # get the averge luminance
            avg = int(getAverage(img))
            # look up the ASCII character for grayscale value(avg)
            if moreLevels:
                gsval = gscale1[int((avg*69)/255)]
            else:
                gsval = gscale2[int((avg*9)/255)]
            aimg[j] += gsval
    # return text image
    return aimg


# def main():
#     # create parser
#     descStr = "this program converts an image into ASCII art"
#     parser = argparse.ArgumentParser(description=descStr)
#     # add expected arguments
#     parser.add_argument('--file', dest='imgFile', required=True)
#     parser.add_argument('--scale', dest='scale', required=False)
#     parser.add_argument('--out', dest='outFile', required=False)
#     parser.add_argument('--cols', dest='cols', required=False)
#     parser.add_argument('--morelevels', dest='morelevels', action='store_true')
#
#     # parse arguments
#     args = parser.parse_args()
#
#     imgFile = args.imgFile
#     # imgFile = 'test.jpg'
#     # set output file
#     outFile = 'out.txt'
#     if args.outFile:
#         outFile = args.outFile
#     # set scale defult as 0.43, which suits a Courier font
#     scale = 0.43
#     if args.scale:
#         scale = float(args.scale)
#     # set cols
#     cols = 80
#     if args.cols:
#         cols = int(args.cols)
#     print("generating ASCII art...")
#     # convert image to ASCII text
#     aimg = covertImageToAscii(imgFile, cols, scale, args.morelevels)
#
#     # open a new text file
#     f = open(outFile, 'w')
#     # write each string in the list to new file
#     for row in aimg:
#         f.write(row + '\n')
#     # clean up
#     f.close()
#     print("ASCII art written to %s" % outFile)
def main():
    imgFile = 'test.jpg'
    outFile = 'out2.txt'
    scale = 0.43
    cols = 200
    print("生成ASCII图像...")
    aimg = covertImageToAscii(imgFile, cols, scale, True)
    f = open(outFile, 'w')
    for row in aimg:
        f.write(row + '\n')
    f.close()
    print("ASCII图像写入到%s" % outFile)


if __name__ == '__main__':
    main()


