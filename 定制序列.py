"""定义一个不可变序列，并记录每个元素的访问次数"""


class countList:
    def __init__(self, *args):  # *args参数为可变数量、类型
        self.values = [x for x in args]
        self.count = {}.fromkeys(range(len(self.values)), 0)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        self.count[key] += 1
        return self.values[key]


c1 = countList(1, 3, 5, 7, 9)
c2 = countList(2, 4, 6, 8, 10)
print(c1[1])
print(c2[1])
print(c1[1] + c2[1])
print(c1.count)
