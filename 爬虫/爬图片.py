import urllib.request
import os


def url_open(url):      # 打开url
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    headers = {'User-Agent': user_agent, }
    request = urllib.request.Request(url, None, headers)  # The assembled request
    response = urllib.request.urlopen(request)
    html = response.read()  # 因为要读取图片，所以不用decode
    print(url)  # 测试代码
    return html


def get_page(url):      # 查找当前页码
    html = url_open(url).decode('utf-8')
    a = html.find('current-comment-page') + 23
    b = html.find(']', a)
    return html[a:b]


def find_imgs(url):     # 寻找图片地址
    html = url_open(url).decode('utf-8')
    img_addrs = []      # 存放图片的列表

    a = html.find('img src=')
    while a != -1:      # 查询本页面所有图片
        b = html.find('.jpg', a, a+255)         # 网页地址最大不超过255
        if b != -1:
            img_addrs.append(html[a+9:b+4])     # 网页地址
        else:
            b = a + 9

        a = html.find('img src=', b)

    for each in img_addrs:
        print(each)



def save_imgs(folder, img_addrs):
    pass


def downloadMM(folder='OOXX', pages = 10):
    os.mkdir(folder)    # 创建目录
    os.chdir(folder)    # 切换目录

    url = 'http://jandan.net/ooxx/'
    page_num = int(get_page(url))

    for i in range(pages):
        page_num -= i
        page_url = url + 'page' + str(page_num) + '#comments'
        img_addrs = find_imgs(page_url)
        save_imgs(folder, img_addrs)


if __name__ == '__main__':
    downloadMM()
