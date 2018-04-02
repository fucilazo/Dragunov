import re


print(re.search(r'apple', 'There are 12 apples on the tree.'))          # 直接查找
print(re.search(r'.', 'There are 12 apples on the tree.'))              # *索引
print(re.search(r'app.', 'There are 12 apples on the tree.'))           # .只能匹配一个字符
print(re.search(r'\.', 'There are 12 apples on the tree.'))             # 普通的'.'
print(re.search(r'[.]', 'There are 12 apples on the tree.'))            # 普通的'.'
print(re.search(r'\d', 'There are 12 apples on the tree.'))             # 普通数字
print(re.search(r'\d\d', 'There are 12 apples on the tree.'))           # 两个连续的普通数字
print(re.search(r'[aeiou]', 'There are 12 apples on the tree.'))        # []里的元素
print(re.search(r'[a-z]', 'There are 12 apples on the tree.'))          # 匹配字母a到z
print(re.search(r'[1-9]', 'There are 12 apples on the tree.'))          # 匹配数字1到9
print(re.search(r'p{2}', 'There are 12 apples on the tree.'))           # 匹配p*2
print(re.search(r'ab{3}c', 'sdgsewrabbbcdfhdfhd'))                      # 匹配a(b*3)c
print(re.search(r'ab{3,10}c', 'sdgsewrabbbbbbbcdfhdfhd'))               # 于abc中匹配3到10次
print(re.search(r'\d{1,5}', '555'))                                     # 1到5位数
print(re.search(r'[0-255]', '188'))                                     # 相当于'[0-2]|5'
print(re.search(r'[01]\d\d|2[0-4]\d|25[0-5]', '188'))                   # 0~255
print(re.search(r'\d\d\d\.\d\d\d\.\d\d\d\.\d\d\d', '192.168.111.123'))  # 三位数IP地址
print(re.search(r'(([01]\d\d|2[0-4]\d|25[0-5])\.){3}([01]\d\d|2[0-4]\d|25[0-5])', '192.168.100.100'))
print(re.search(r'(([01]?\d?\d|2[0-4]\d|25[0-5])\.){3}([01]?\d?\d|2[0-4]\d|25[0-5])', '192.168.12.1'))
print('----------------------------*分*隔*符*----------------------------')
print(re.search(r'Fuci(l|o)', 'Fucilazo'))      # '或'匹配
print(re.search(r'^Fuci', 'Fucilazo'))          # 检测是否处于字符串起始位置，若不是则返回None
print(re.search(r'lazo$', 'Fucilazo'))          # 检测是否处于字符串结尾位置，若不是则返回None
print(re.search(r'abc*', 'ab'))                 # 等价于{0,}
print(re.search(r'abc+', 'abc'))                # 等价于{1,}
print(re.search(r'abc?', 'ab'))                 # 匹配一个字符1次或0次   以上这三个要优于{}，因为python内部会对其进行优化
print('----------------------------*分*隔*符*----------------------------')
print(re.findall(r'[a-z]', 'Fucilazo'))         # 找到所有符合的字符
print(re.findall(r'[A-z]', 'Fucilazo'))         # 找到所有符合的字符
print(re.findall(r'[^a-z]', 'FuciLazo'))        # 取反
print(re.findall(r'[a-z^]', 'FuciLazo^'))       # 匹配[a-z]和'^'
print('----------------------------*分*隔*符*----------------------------')
s = '<html><title>http://juqing.9duw.com</title></html>'
print(re.search(r'<.+>', s))    # 贪婪模式
print(re.search(r'<.+?>', s))   # 启用非贪婪（*?、+?、??）

# test