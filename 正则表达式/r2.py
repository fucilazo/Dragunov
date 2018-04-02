"""
如果要重复的使用某个正则表达式，
那么可以将该正则表达式编译成模式对象
使用re.compile()方法来编译
"""
import re


p = re.compile(r'[A-Z]')
print(type(p))

print(p.search('there have a BIG apple!'))
print(p.findall('there have a BIG apple!'))


charref1 = re.compile(r'\d+\.\d*')
charref2 = re.compile(r'''
                        \d +    # the integral part
                        \.      # the decimal point
                        \d *    # some fractional digits''', re.X)
print(charref1.findall('π=3.1415926'))
print(charref2.findall('π=3.1415926'))
