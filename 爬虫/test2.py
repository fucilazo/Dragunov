import urllib.request

user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

url = 'http://juqing.9duw.com/yiyuele/news/66187.html'
headers = {'User-Agent': user_agent, }

request = urllib.request.Request(url, None, headers)    # The assembled request
response = urllib.request.urlopen(request)
html = response.read().decode('ANSI')  # The data u need

# a = html.find('current-comment-page') + 23
# b = html.find(']', a)
#
# print(html[a:b])    # 打印当前页码
print(html)
