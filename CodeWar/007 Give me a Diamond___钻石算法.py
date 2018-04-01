"""This kata is to practice simple string output. Jamie is a programmer, and James' girlfriend. She likes diamonds, and wants a diamond string from James. Since James doesn't know how to make this happen, he needs your help.

###Task:

You need to return a string that displays a diamond shape on the screen using asterisk ("*") characters. Please see provided test cases for exact output format.

The shape that will be returned from print method resembles a diamond, where the number provided as input represents the number of *’s printed on the middle line. The line above and below will be centered and will have 2 less *’s than the middle line. This reduction by 2 *’s for each line continues until a line with a single * is printed at the top and bottom of the figure.

Return null if input is even number or negative (as it is not possible to print diamond with even number or negative number).

Please see provided test case(s) for examples.

Python Note

Since print is a reserved word in Python, Python students must implement the diamond(n) method instead, and return None for invalid input.

JS Note

JS students, like Python ones, must implement the diamond(n) method, and return null for invalid input."""


def diamond(n):
    expected = ''
    if n%2 == 0 or n <= 0:
        return None
    for i in range(0, int((n-1)/2)):
        space = int((n-1-i*2)/2)
        star = 1+i*2
        # for j in range(0, space):
        #     print(' ', end='')
        # for j in range(0, star):
        #     print('*', end='')
        # for j in range(0, space):
        #     print(' ', end='')
        # print('')
        for j in range(0, space):
            expected += ' '
        for j in range(0, star):
            expected += '*'
        for j in range(0, space):
            expected += ' '
        expected += '\n'
    for i in range(0, n):
    #     print('*', end='')
    # print('')
        expected += '*'
    expected += '\n'
    for i in range(0, int((n-1)/2)):
        star = n-2*(i+1)
        space = i+1
        # for j in range(0, space):
        #     print(' ', end='')
        # for j in range(0, star):
        #     print('*', end='')
        # for j in range(0, space):
        #     print(' ', end='')
        # print('')
        for j in range(0, space):
            expected += ' '
        for j in range(0, star):
            expected += '*'
        for j in range(0, space):
            expected += ' '
        expected += '\n'
    return expected


#best
def best(n):
    if n > 0 and n % 2 == 1:
        diamond = ""
        for i in range(n):
            diamond += " " * abs((n/2) - i)
            diamond += "*" * (n - abs((n-1) - 2 * i))
            diamond += "\n"
        return diamond
    else:
        return None
