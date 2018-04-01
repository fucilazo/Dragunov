"""The goal of this exercise is to convert a string to a new string where each character in the new string is '(' if that character appears only once in the original string, or ')' if that character appears more than once in the original string. Ignore capitalization when determining if a character is a duplicate.

Examples:

"din" => "((("

"recede" => "()()()"

"Success" => ")())())"

"(( @" => "))(("

"""


def duplicate_encode(word):
    result = ''
    word = word.lower()
    for i in word:
        if word.count(i) > 1:
            result += ')'
        else:
            result += '('
    return result


# best
def best(word):
    return "".join(["(" if word.lower().count(c) == 1 else ")" for c in word.lower()])
def best2(word):
    word = word.lower()
    return ''.join('(' if word.count(x) == 1 else ')' for x in word)
