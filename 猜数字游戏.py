import  random

print("--------------G-A-M-E--------------")
rightNum = random.randint(1, 9)
temp = input("Plz input a num:")
guess = int(temp)
i = 0
while guess != rightNum and i < 4:
    i = i+1
    if guess > rightNum:
        print("It's BIG!!")
        temp = input("Plz input a num:")
        guess = int(temp)
    else:
        print("It's SMALL!!")
        temp = input("Plz input a num:")
        guess = int(temp)
if i == 4:
    print("Time up")
else:
    print("U R Right")
print("----------------END----------------")
input("Press <enter>")