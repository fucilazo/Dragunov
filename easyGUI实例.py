import easygui as g
import sys

while 1:
    g.msgbox("Hey,welcome to easyGUI game")

    msg = "What do u want？"
    title = "MiniGame"
    choices = ["Apple", "Peach", "Money"]

    choice = g.choicebox(msg, title, choices)

    # note that we convert choice to string, in case
    # the user cancelled the choice, and we got None.
    g.msgbox("Your choice is: " + str(choice), "Result")

    msg = "Do u want to replay？"
    title = "U choose"

    if g.ccbox(msg, title):  # show a Continue/Cancel dialog
        pass  # user chose Continue
    else:
        sys.exit(0)  # user chose Cancel
