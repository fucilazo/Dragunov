"""There is an array with some numbers. All numbers are equal except for one. Try to find it!

findUniq([ 1, 1, 1, 2, 1, 1 ]) === 2
findUniq([ 0, 0, 0.55, 0, 0 ]) === 0.55
It’s guaranteed that array contains more than 3 numbers.

The tests contain some very huge arrays, so think about performance."""


# def find_uniq(arr):  此算法在codewar上无法运行!
#     for i in arr:
#         if arr.count(i) is 1:
#             return i
def find_uniq(arr):
    arr = sorted(arr)
    if arr[0] is arr[1]:
        arr = sorted(arr, reverse=True)
        return arr[0]
    else:
        return arr[0]


# best
def best(arr):
    a, b = set(arr)
    return a if arr.count(a) == 1 else b

def another_solution(arr):
    arr.sort()
    return arr[0] if arr[0] != arr[1] else arr[-1]

def solution2(arr):
    return [x for x in set(arr) if arr.count(x) == 1][0]
