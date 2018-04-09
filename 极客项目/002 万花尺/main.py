import sys
import random
import argparse
import numpy as py
import math
import turtle
from PIL import Image
from datetime import datetime
from fractions import gcd


# A class that draw a Spirograph
class Spiro:
    def __init__(self, xc, yc, col, R, r, l):
        self.t = turtle.Turtle()    # 创建turtle对象
        self.t.shape('turtle')      # 'turtle'形状的游标
        self.step = 5               # set the step in degree
        self.drawingComplete = False
        self.setparams(xc, yc, col, R, r, l)
        self.restart()

    # 参数
    def setparams(self, xc, yc, col, R, r, l):
        self.xc = xc
        self.yc = yc
        self.R = int(R)
        self.r = int(r)
        self.l = l
        self.col = col
        gcdVal = gcd(self.r, self.R)    # reduce r/R to its smallest from by dividing with the GCD
        self.nRot = self.r//gcdVal
        self.k = r/float(R)             # get ratio of radii
        self.t.color(*col)              # set color
        self.a = 0                      # store the current angle

    # 重新绘图
    def restart(self):
        self.drawingComplete = False    # set the flag
        self.t.showturtle()             # show the turtle
        # 定位到起始点
        self.t.up()
        R, k, l = self.R, self.k, self.l
        a = 0.0
        x = R*((1-k)*math.cos(a) + 1*k*math.cos((1-k)*a/k))
        y = R*((1-k)*math.sin(a) + 1*k*math.sin((1-k)*a/k))
        self.t.setpos(self.xc + x, self.yc + y)
        self.t.down()

    # Draw the whole thing
    def draw(self):     # drawing the rest of points
        R, k, l = self.R, self.k, self.l
        for i in range(0, 360*self.nRot + 1, self.step):
            a = math.radians(i)
            x = R * ((1 - k) * math.cos(a) + 1 * k * math.cos((1 - k) * a / k))
            y = R * ((1 - k) * math.sin(a) + 1 * k * math.sin((1 - k) * a / k))
            self.t.setpos(self.xc + x, self.yc + y)
            self.t.hideturtle()     # 绘制完成，隐藏cursor

        # update by one step
    def update(self):
        if self.drawingComplete:    # skip the rest of the steps if done
            return
        self.a += self.step     # increment the angle
        R, k, l = self.R, self.k, self.l    # draw a step
        a = math.radians(self.a)    # set the angle
        x = self.R * ((1 - k) * math.cos(a) + 1 * k * math.cos((1 - k) * a / k))
        y = self.R * ((1 - k) * math.sin(a) + 1 * k * math.sin((1 - k) * a / k))
        self.t.setpos(self.xc + x, self.yc + y)
        if self.a >= 360*self.nRot:     # if drawing is complete, set the flag
            self.drawingComplete = True
            self.t.hideturtle()     # drawing is now done, so hide the cursor

    # clear everything
    def clear(self):
        self.t.clear()


# A class for animating Spirographs
class SpiroAnimator:
    def __init__(self, N):
        self.deltaT = 10    # set the timer value in milliseconds
        # get the window dimensions
        self.width = turtle.window_width()
        self.height = turtle.window_height()
        self.spiros = []    # create the Spiro objects
        for i in range(N):
            rparams = self.genRandomParams()    # 产生随机数据
            spiro = Spiro(*rparams)     # set the spiro parameters
            self.spiros.append(spiro)
        turtle.ontimer(self.update, self.deltaT)    # call timer

    # restart spiro drawing
    def restart(self):
        for spiro in self.spiros:
            spiro.clear()   # clear
            rparams = self.genRandomParams()    # 产生随机数据
            spiro = Spiro(*rparams)     # set the spiro parameters
            spiro.restart()     # restart drawing

    # generate random parameters
    def genRandomParams(self):
        width, height = self.width, self.height
        R = random.randint(50, min(width, height)//2)
        r = random.randint(10, 9*R//10)
        l = random.uniform(0.1, 0.9)
        xc = random.randint(-width//2, width//2)
        yc = random.randint(-height//2, height//2)
        col = (random.random(), random.random(), random.random())
        return (xc, yc, col, R, r, l)

    def update(self):
        # update all spiros
        nComplete = 0
        for spiro in self.spiros:
            spiro.update()  # update
            if spiro.drawingComplete:   # count complete spiros
                nComplete += 1
        if nComplete == len(self.spiros):   # restart if all spiros are complete
            self.restart()
        turtle.ontimer(self.update, self.deltaT)    # call the timer

    # toggle turtle cursor on and off
    def toggleTurtles(self):
        for spiro in self.spiros:
            if spiro.t.isvisible():
                spiro.t.hideturtle()
            else:
                spiro.t.showturtle()


# save drawing as PNG files
def saveDrawing():
    # hide turtle
    turtle.hideturtle()
    # generate unique file name
    dateStr = (datetime.now()).strftime("%d%b%Y-%H%M%S")
    fileName = 'spiro-' + dateStr
    print('saving drawing to %s.eps/png' % fileName)
    # get tkinter canvas
    canvas = turtle.getcanvas()
    # save postscipt image
    canvas.postscript(file=fileName + '.eps')
    # use PIL to convert to PNG
    img = Image.open(fileName + '.eps')
    img.save(fileName + '.png', 'png')
    # show turtle
    turtle.showturtle()


# main函数
def main():
    print('generating spirograph...')   # use sys.argv if needed
    # create parser
    descStr = """This program draws spirographs using the Turtle module. 
    When run with no arguments, this program draws random spirographs.

    Terminology:
    R: radius of outer circle.
    r: radius of inner circle.
    l: ratio of hole distance to r.
    """

    parser = argparse.ArgumentParser(description=descStr)

    # add expected arguments
    parser.add_argument('--sparams', nargs=3, dest='sparams', required=False, help="R,r,l")

    # parse args
    args = parser.parse_args()

    turtle.setup(width=0.8)         # set the width of the drawing window to 80 percent of the screen width
    turtle.shape('turtle')          # set the cursor shape to tuetle
    turtle.title("Spirographs")     # set title
    turtle.onkey(saveDrawing, "s")  # add the key handler to save our drawings
    turtle.listen()                 # start listening
    turtle.hideturtle()             # hide the main turtle cursor

    # check for ny arguments sent to --sparams and draw the Spirograph
    if args.sparams:
        params = [float(x) for x in args.sparams]
        # draw spirograph with given parameters
        # black by default
        col = (0.0, 0.0, 0.0)
        spiro = Spiro(0, 0, col, *params)
        spiro.draw()
    else:
        spiroAnim = SpiroAnimator(4)                # create the animator object
        turtle.onkey(spiroAnim.toggleTurtles, "t")  # add a key handler to toggle the turtle cursor
        turtle.onkey(spiroAnim.restart, "space")    # add a key handler to restart the animation

    # start the turtle main loop
    turtle.mainloop()


# call main
if __name__ == '__main__':
    main()
