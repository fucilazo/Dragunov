"""Given an array of one's and zero's convert the equivalent binary value to an integer.

Eg: [0, 0, 0, 1] is treated as 0001 which is the binary representation of 1

Examples:

Testing: [0, 0, 0, 1] ==> 1
Testing: [0, 0, 1, 0] ==> 2
Testing: [0, 1, 0, 1] ==> 5
Testing: [1, 0, 0, 1] ==> 9
Testing: [0, 0, 1, 0] ==> 2
Testing: [0, 1, 1, 0] ==> 6
Testing: [1, 1, 1, 1] ==> 15
Testing: [1, 0, 1, 1] ==> 11"""


def binary_array_to_number(arr):
    s = 0
    j = len(arr)-1
    for i in arr:
        if i:
            s += pow(2, j)
        j -= 1
    return s


# best
def best(arr):
    return int("".join(map(str, arr)), 2)   # int()函数转换进制：int(*str, num)


# solution_2
def solution_2(arr):
    s = 0
    for digit in arr:
        s = s * 2 + digit
    return s
