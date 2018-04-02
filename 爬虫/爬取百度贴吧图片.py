import urllib.request
import re


def open_url(url):
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    headers = {'User-Agent': user_agent, }
    request = urllib.request.Request(url, None, headers)  # The assembled request
    response = urllib.request.urlopen(request)
    html = response.read().decode('utf-8')
    return html


def get_img(html):
    # p = r'<img class="BDE_Image" src="[^"]+\.jpg"'
    p = r'<img class="BDE_Image" src="([^"]+\.jpg)"'
    # If one or more groups are present in the pattern, return a list of groups;
    # this will be a list of tuples if the pattern has more than one group.
    # Empty matches are included in the result unless they touch the beginning of another match.
    # 如果含有小括号，会自动将小括号内的元素返回；若含有多个括号，则返回一个元组
    imglist = re.findall(p, html)

    for each in imglist:
        filename = each.split('/')[-1]
        urllib.request.urlretrieve(each, filename)


if __name__ == '__main__':
    url = 'https://tieba.baidu.com/p/5453452904'
    get_img(open_url(url))
