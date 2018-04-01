import urllib.request
import random

url = 'http://www.whatismyip.com.tw'
#url = 'http://www.baidu.com'

iplist = ['119.28.152.208:80', '114.215.95.188:3128', '117.68.195.202:18118']

proxy_support = urllib.request.ProxyHandler({'http': random.choice(iplist)})

opener = urllib.request.build_opener(proxy_support)
opener.add_handler = [('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.104 Safari/537.36 Core/1.53.4882.400 QQBrowser/9.7.13059.400')]

urllib.request.install_opener(opener)

response = urllib.request.urlopen(url)
html = response.read().decode('utf-8')

print(html)
