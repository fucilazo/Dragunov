import urllib.request
import os


def url_open(url):      # 打开url
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    headers = {'User-Agent': user_agent, }
    request = urllib.request.Request(url, None, headers)  # The assembled request
    response = urllib.request.urlopen(request)
    html = response.read()  # 因为要读取图片，所以不用decode
    return html


def find_imgs(url):     # 寻找图片地址
    html = url_open(url).decode('ANSI')
    img_addrs = []      # 存放图片的列表

    a = html.find('/UploadPic')
    while a != -1:      # 查询本页面所有图片
        b = html.find('.jpg', a, a+255)         # 网页地址最大不超过255
        if b != -1:
            img_addrs.append('http://juqing.9duw.com' + html[a:b+4])     # 网页地址
        else:
            b = a

        a = html.find('/UploadPic', b)

    return img_addrs


def save_imgs(folder, img_addrs):
    for each in img_addrs:
        filename = each.split('/')[-1]
        with open(filename, 'wb') as f:
            img = url_open(each)
            f.write(img)


def downloadMM(folder='OOXX', pages=10):
    os.mkdir(folder)    # 创建目录
    os.chdir(folder)    # 切换目录

    url = 'http://juqing.9duw.com/yiyuele/news/66187'

    for i in range(1, pages):
        if i == 1:
            page_url = url + '.html'
        else:
            page_url = url + '_' + str(i) + '.html'
        img_addrs = find_imgs(page_url)
        save_imgs(folder, img_addrs)


if __name__ == '__main__':
    downloadMM()
