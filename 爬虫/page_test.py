url = 'http://juqing.9duw.com/yiyuele/news/66187'

for i in range(1, 10):
    if i == 1:
        page_url = url + '.html'
    else:
        page_url = url + '_' + str(i) + '.html'
    print(page_url)
