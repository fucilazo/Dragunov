import urllib.request
import os


def url_open(url):      # 打开url
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    headers = {'User-Agent': user_agent, }
    request = urllib.request.Request(url, None, headers)  # The assembled request
    response = urllib.request.urlopen(request)
    html = response.read()
    return html


def find_imgs(url):     # 寻找图片地址
    html = url_open(url).decode('utf-8')
    img_addrs = []      # 存放图片的列表
    a = html.find('https://gg.385gg.com')
    print(url)
    while a != -1:      # 查询本页面所有图片
        b = html.find('.gif', a, a + 255)  # 网页地址最大不超过255
        if b != -1:
            img_addrs.append(html[a:b + 4])  # 网页地址
        else:
            b = a
        a = html.find('https://gg.385gg.com', b)
    # for each in img_addrs:
    #     print(each)
    return img_addrs    # 本页GIF数组(x8)


def save_imgs(img_addrs):
    for each in img_addrs:
        filename = each[25:]
        filename = filename.replace('/', '.')
        with open(filename, 'wb') as f:
            img = url_open(each)
            f.write(img)


def downloadMM(folder='GIF', pages=1130):
    # os.mkdir(folder)    # 创建目录
    os.chdir(folder)    # 切换目录
    url = 'https://www.868cf.com/htm/gif0/'
    for i in range(1073, pages+1):
        page_url = url + str(i) + '.htm'
        find_imgs(page_url)
        img_addrs = find_imgs(page_url)
        save_imgs(img_addrs)


if __name__ == '__main__':
    downloadMM()
