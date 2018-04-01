"""Snail Sort

Given an n x n array, return the array elements arranged from outermost elements to the middle element, traveling clockwise.

array = [[1,2,3],
         [4,5,6],
         [7,8,9]]
snail(array) #=> [1,2,3,6,9,8,7,4,5]
For better understanding, please follow the numbers of the next array consecutively:

array = [[1,2,3],
         [8,9,4],
         [7,6,5]]
snail(array) #=> [1,2,3,4,5,6,7,8,9]
This image will illustrate things more clearly:



NOTE: The idea is not sort the elements from the lowest value to the highest; the idea is to traverse the 2-d array in a clockwise snailshell pattern.

NOTE 2: The 0x0 (empty matrix) is represented as [[]]
"""


def snail(array):
    result = []
    n = len(array)

    if not array[0]:
        return result
    if n == 1:
        result.append(array[0][0])
        return result

    for i in range(n):
        for j in range(i, n-1-i):
            result.append(array[i][j])      # UP
        for j in range(i, n-1-i):
            result.append(array[j][-1-i])    # RIGHT
        for j in range(n-1-i, i, -1):
            result.append(array[-1-i][j])    # DOWN
        for j in range(n-1-i, i, -1):
            result.append(array[j][i])      # LEFT
        if n%2 == 1 and i == int(n/2)-1:
            result.append(array[int(n/2)][int(n/2)])
            return result
        elif n%2 == 0 and i == (n/2)-1:
            return result


# print(snail([[1, 2, 3, 4, 5],
#              [6, 7, 8, 9, 10],
#              [11,12,13,14,15],
#              [16,17,18,19,20],
#              [21,22,23,24,25]]))
# print(snail([[1, 2, 3, 4],
#              [6, 7, 8, 9],
#              [11,12,13,14],
#              [16,17,18,19],]))
print(snail([[1]]))


# Clever
def clever(array):
    return list(array[0]) + snail(zip(*array[1:])[::-1]) if array else []
# 2
def clever_2(array):
    a = []
    while array:
        a.extend(list(array.pop(0)))
        array = zip(*array)
        array.reverse()
    return a