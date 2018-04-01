class New_int(int):     # 可以写成 class int(int) 从而替换int
    def __add__(self, other):
        return int.__sub__(self, other)
    def __sub__(self, other):
        return int.__add__(self, other)


a1 = New_int(3)
b1 = New_int(5)
print('a+b=', a1+b1)
print('a-b=', a1-b1)


class Try_int(int):
    def __add__(self, other):
        return int(self) + int(other)
    def __sub__(self, other):
        return int(self) - int(other)


a2 = Try_int(3)
b2 = Try_int(5)
print('a+b=', a2+b2)
print('a-b=', a2-b2)
