"""Build Tower by the following given arguments:
number of floors (integer and always greater than 0)
block size (width, height) (integer pair and always greater than (0, 0))

Tower block unit is represented as *

Python: return a list;
JavaScript: returns an Array;
Have fun!

for example, a tower of 3 floors with block size = (2, 3) looks like below

[
  ['    **    '],
  ['    **    '],
  ['    **    '],
  ['  ******  '],
  ['  ******  '],
  ['  ******  '],
  ['**********'],
  ['**********'],
  ['**********']
]
and a tower of 6 floors with block size = (2, 1) looks like below

[
  '          **          ',
  '        ******        ',
  '      **********      ',
  '    **************    ',
  '  ******************  ',
  '**********************'
]"""


def tower_builder(n, block_size):
    w, h = block_size
    tower = []
    for i in range(n):
        for j in range(h):
            tower.append(' '*(w*(n-i-1)) + '*'*(w*(2*i+1)) + ' '*(w*(n-i-1)))
    return tower


# solution
def solution(n, block_size):
    w, h = block_size
    return [str.center("*" * (i*2-1)*w, (n*2-1)*w) for i in range(1, n+1) for _ in range(h)]

print(tower_builder(3, (4, 2)))
