"""
1.右键网页or打开开发者工具
2.点击Network一栏
3.输入英文，翻译。查找post或其他提交数据的网页行为
4.查阅General和RequestHeader
"""

import urllib.request
import urllib.parse
import json
import time

while True:
    content = input('请输入需要翻译的内容(输入"q!"退出程序)：')
    if content == 'q!':
        break

    # url = 'http://fanyi.youdao.com/translate_o?smartresult=dict&smartresult=rule'
    url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule'

    head = {}
    head['User-Agent'] = 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.104 Safari/537.36 Core/1.53.4882.400 QQBrowser/9.7.13059.400'

    data = {}
    data['i'] = content     # 不能去！
    data['from'] = 'AUTO'
    data['to'] = 'AUTO'
    data['smartresult'] = 'dict'
    data['client'] = 'fanyideskweb'
    data['salt'] = '1522250255734'
    data['sign'] = '441942c8c597c0b1a51fc12fd0ba794e'
    data['doctype'] = 'json'        # 不能去！
    data['version'] = '2.1'
    data['keyfrom'] = 'fanyi.web'
    data['action'] = 'FY_BY_CLICKBUTTION'
    data['typoResult'] = 'false'
    data = urllib.parse.urlencode(data).encode('utf-8')

    req = urllib.request.Request(url, data, head)
    # req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.104 Safari/537.36 Core/1.53.4882.400 QQBrowser/9.7.13059.400')

    response = urllib.request.urlopen(req)
    html = response.read().decode('utf-8')     # windos下不必要decode

    print(html)

    target = json.loads(html)
    print(target)
    print(target['translateResult'])
    # for i in target['translateResult'][0]:
    #     print(i, end='')
    for i in range(len(target['translateResult'][0])):
        print(target['translateResult'][0][i], end='')
    print()
    for i in range(len(target['translateResult'][0])):
        print(target['translateResult'][0][i]['tgt'], end='')
    print()
    # print(target['translateResult'][0][0])
    # print(target['translateResult'][0][0]['tgt'])
    time.sleep(5)

