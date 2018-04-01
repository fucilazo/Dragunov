"""I am the one who establishes the list so I told him: "Don't worry any more, I will modify the order of the list". It was decided to attribute a "weight" to numbers. The weight of a number will be from now on the sum of its digits.

For example 99 will have "weight" 18, 100 will have "weight" 1 so in the list 100 will come before 99. Given a string with the weights of FFC members in normal order can you give this string ordered by "weights" of these numbers?

Example:

"56 65 74 100 99 68 86 180 90" ordered by numbers weights becomes: "100 180 90 56 65 74 68 86 99"

When two numbers have the same "weight", let us class them as if they were strings and not numbers: 100 is before 180 because its "weight" (1) is less than the one of 180 (9) and 180 is before 90 since, having the same "weight" (9) it comes before as a string.

All numbers in the list are positive numbers and the list can be empty."""

string = "2000 10003 1234000 44444444 9999 11 11 22 123"
def order_weight(string):
    list = ''
    string = string.split()
    string = sorted(string, key=(lambda x: (fun(x), fun2(x), fun3(x), fun4(x), fun5(x), fun6(x), len(x))))
    for i in string:
        list += (i+' ')
    return list.rstrip()
def fun(a):
    a = list(a)
    s = 0
    for i in a:
        s += int(i)
    return s
def fun2(a):
    a = list(a)
    return a[0]
def fun3(a):
    if(len(a)>1):
        a = list(a)
        return a[1]
def fun4(a):
    if (len(a) > 2):
        a = list(a)
        return a[2]
def fun5(a):
    if (len(a) > 3):
        a = list(a)
        return a[3]
def fun6(a):
    if (len(a) > 4):
        a = list(a)
        return a[4]
print(order_weight(string))


# best
def best(_str):
    return ' '.join(sorted(sorted(_str.split(' ')), key=lambda x: sum(int(c) for c in x)))
# best
order_weight=lambda s:' '.join(sorted(s.split(),key=lambda s:(sum(map(int,s)),s)))