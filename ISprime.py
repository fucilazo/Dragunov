import time
# import math
# def normal(n):
#     time.clock()
#     count = 0
#     for i in range(2, n):
#         flag = 0
#         if i>2 and i%2 == 0:
#             continue
#         elif i>5 and i%5 == 0:
#             continue
#         for j in range(2, int(math.sqrt(i))+1):
#             if i % j == 0:
#                 flag = 1
#         if flag == 0:
#             count += 1
#             print(i)
#     print(count)
#     print(time.clock())
# normal(1000000)


# def odd_iter():
#     n = 3
#     while True:
#         yield n
#         n += 2
# def not_divisible(n):
#     return lambda x: x % n != 0
# def prime():
#     yield 2
#     it = odd_iter()
#     while True:
#         n = next(it)
#         yield n
#         it = filter(not_divisible(n), it)
# for i in prime():
#     if i < 100000:
#         print(i)
#         print(time.clock())
#     else:
#         break



# def prime(n):
#     lis = {}
#     for a in range(2, n+1):
#         if a not in lis:
#             lis = 1
#             k = a*2
#             while k <= n:
#                 lis[k] = 0
#                 k = k+a
#     ans = []
#     for b in lis:
#         if lis == 1:
#             ans.append(b)
#     return ans
# for i in prime(10000):
#     print(i)
#     print(time.clock())


def eladuosai(n):
    time.clock()
    arr = list(range(1, n+1))
    arr[0] = 0
    for i in range(2, n+1):
        if arr[i-1] != 0:
            for j in range(i*2, n+1, i):
                arr[j-1] = 0
    result = [x for x in arr if x != 0]
    return result
for j in eladuosai(100):
    print(j)
print(time.clock())

