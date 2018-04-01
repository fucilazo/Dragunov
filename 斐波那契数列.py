#黄金分割比
def fun(x):
    if x == 1:
        return 1
    elif x == 2:
        return 1
    else:
        return fun(x-1)+fun(x-2)
x = int(input("请输入斐波那契数列中组数："))
print("该组的值为：%d"%fun(x))
print("分割比为：%f"%((fun(x-1))/(fun(x))))