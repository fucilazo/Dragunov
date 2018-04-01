# 要么怎样，要么不怎样
if A:
    pass
else:
    pass


# 干完了能怎样，干不完别想怎样(while, for)
def showMaxFactor(num):
    count = num // 2    # 向下取整
    while count > 1:
        if num % count == 0:
            print('%d最大的约数是%d' % (num, count))
            break
        count -= 1
    else:
        print('%d是素数' % num)


num = int(input('请输入一个数:'))
showMaxFactor(num)


# 没有问题，那就干吧
try:
    int('abc')
except ValueError as reason:
    print('出错啦' + str(reason))
else:
    print('没有任何异常')
