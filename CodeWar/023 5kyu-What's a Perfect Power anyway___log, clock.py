"""A perfect power is a classification of positive integers:

In mathematics, a perfect power is a positive integer that can be expressed as an integer power of another positive integer. More formally, n is a perfect power if there exist natural numbers m > 1, and k > 1 such that mk = n.
Your task is to check wheter a given integer is a perfect power. If it is a perfect power, return a pair m and k with mk = n as a proof. Otherwise return Nothing, Nil, null, NULL, None or your language's equivalent.

Note: For a perfect power, there might be several pairs. For example 81 = 3^4 = 9^2, so (3,4) and (9,2) are valid solutions. However, the tests take care of this, so if a number is a perfect power, return any pair that proves it.

Examples

isPP(4) => [2,2]
isPP(9) => [3,2]
isPP(5) => None"""

from random import random, randrange
from math import log, floor
from time import *

def isPP(n):
    for i in range(2, int(n**0.5)+2):       # Note:i**j waste much more time than j**i
        for j in range(2, int(n**0.5)+2):
            if j**i == n:
                return [j, i]
    return None


for i in range(10):
    m = 2 + floor(random() * 255)
    k = 2 + floor(random() * log(268435455) / log(m))
    l = m**k
    print(l, isPP(l), clock())


# best
from math import ceil, log, sqrt
def isPP(n):
    for b in range(2, int(sqrt(n)) + 1):
        e = int(round(log(n, b)))
        if b ** e == n:
            return [b, e]
    return None

