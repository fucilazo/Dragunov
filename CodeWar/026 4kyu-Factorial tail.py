"""The problem

How many zeroes are at the end of the factorial of 10? 10! = 3628800, i.e. there are 2 zeroes. 16! in hexadecimal would be 0x130777758000, which has 3 zeroes.

Scalability

Unfortunately, machine integer numbers has not enough precision for larger values. Floating point numbers drop the tail we need. We can fall back to arbitrary-precision ones - built-ins or from a library, but calculating the full product isn't an efficient way to find just the tail of a factorial. Calculating 100'000! in compiled language takes around 10 seconds. 1'000'000! would be around 10 minutes, even using efficient Karatsuba algorithm

Your task

is to write a function, which will find the number of zeroes at the end of (number) factorial in arbitrary radix = base for larger numbers.

base is an integer from 2 to 256
number is an integer from 1 to 1'000'000"""

import re


def zeroes(base, number):
    f = 1
    for i in range(number): f *= i
    m = re.search("0+$", str(f))
    return m.end(0) - m.start(0) + 1