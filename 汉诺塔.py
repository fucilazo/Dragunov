def hanoi(n, x, y, z):
    global i
    if n == 1:
        print(x, '-->', z)
        # i += 1
    else:
        hanoi(n-1, x, z, y)  # 将前n-1个盘子从x移动到y上
        i += 1
        print(x, '-->', z)  # 将最底下的最后一个盘子从x移动到z上
        hanoi(n-1, y, x, z)  # 将y上的n-1个盘子移动到z上
        i += 1


i = 0
n = int(input("请输入汉诺塔层数："))
hanoi(n, 'X', 'Y', 'Z')
print("共移动了%d次" % (i+1))
