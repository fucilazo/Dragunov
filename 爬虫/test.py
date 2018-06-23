import urllib.request

user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

url = 'https://www.868cf.com/htm/gif0/463.htm'
headers = {'User-Agent': user_agent, }

request = urllib.request.Request(url, None, headers)    # The assembled request
response = urllib.request.urlopen(request)
html = response.read().decode('utf-8')  # The data u need
img_addrs = []      # 存放图片的列表

a = html.find('https://gg.385gg.com')
while a != -1:      # 查询本页面所有图片
    # print(url)
    b = html.find('.gif', a, a+255)         # 网页地址最大不超过255
    if b != -1:
        img_addrs.append(html[a:b+4])     # 网页地址
    else:
        b = a
    # print(html[a:b+4])

    a = html.find('https://gg.385gg.com', b)


for each in img_addrs:
    print(each)


