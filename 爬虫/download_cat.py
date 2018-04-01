import urllib.request

# response = urllib.request.urlopen('http://placekitten.com/g/600/500')
req = urllib.request.Request('http://placekitten.com/g/600/500')
response = urllib.request.urlopen(req)

cat_img = response.read()

with open('cat_600_500.jpg', 'wb') as f:
    f.write(cat_img)

print(response.geturl())
print(response.info())
print(response.getcode())   # 200表示正常响应
