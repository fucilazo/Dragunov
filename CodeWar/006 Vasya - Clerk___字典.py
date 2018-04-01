"""The new "Avengers" movie has just been released! There are a lot of people at the cinema box office standing in a huge line. Each of them has a single 100, 50 or 25 dollars bill. A "Avengers" ticket costs 25 dollars.

Vasya is currently working as a clerk. He wants to sell a ticket to every single person in this line.

Can Vasya sell a ticket to each person and give the change if he initially has no money and sells the tickets strictly in the order people follow in the line?

Return YES, if Vasya can sell a ticket to each person and give the change. Otherwise return NO.

###Examples:

### Python ###

tickets([25, 25, 50]) # => YES
tickets([25, 100])
         # => NO. Vasya will not have enough money to give change to 100 dollars"""


def tickets(people):
    vasya = {25:0, 50:0}
    if people[0] != 25:
        return "NO"
    for i in people:
        if i == 25:
            vasya[25] += 1
        elif i == 50 and vasya[25] > 0:
            vasya[25] -= 1
            vasya[50] += 1
        elif i == 100:
            if vasya[50] > 0 and vasya[25] > 0:
                vasya[50] -= 1
                vasya[25] -= 1
            elif vasya[25] > 2:
                vasya[25] -= 3
            else: return "NO"
        else:
            return "NO"
    return "YES"


# clever
def clever(people):
    till = {100.0: 0, 50.0: 0, 25.0: 0}

    for paid in people:
        till[paid] += 1
        change = paid - 25.0

        for bill in (50, 25):
            while bill <= change and till[bill] > 0:
                till[bill] -= 1
                change -= bill

        if change != 0:
            return 'NO'

    return 'YES'
