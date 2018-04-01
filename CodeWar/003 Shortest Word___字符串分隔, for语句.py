"""x Simple, given a string of words, return the length of the shortest word(s).

String will never be empty and you do not need to account for different data types."""


def find_short(s):
    s += ' '
    l = 0
    m = []
    for i in s:
        if i == ' ':
            m.append(l)
            l = -1
        l += 1
    return min(m)


# best
def best_1(s):
    return min(len(x) for x in s.split())   # 'for' sentense
def best_2(s):
    return len(min(s.split(' '), key=len))
def best_3(s):
    s = s.split()  # splits the string into a list of individual words
    l = min(s, key = len)  # finds the shortest string in the list
    return len(l) # returns shortest word length
