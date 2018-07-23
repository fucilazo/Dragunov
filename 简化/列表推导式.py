numbers = [1, 2, 3, 4, 5]

double_odds = []
for n in numbers:
    if n%2 == 1:
        double_odds.append(n*2)
print(double_odds)
# <<<<<简化>>>>>
double_odds = [n*2 for n in numbers if n%2 == 1]
print(double_odds)
# ===========================================================
x = [1, 2, 3, 4]
out = []
for i in x:
    out.append(i**2)
print(out)
# <<<<<简化>>>>>
out = [i**2 for i in x]
print(out)
# ===========================================================
multiples = (i for i in range(30) if i % 3 is 0)    # 产生一个生成器
print(multiples)
print(multiples.__next__())
print(multiples.__next__())