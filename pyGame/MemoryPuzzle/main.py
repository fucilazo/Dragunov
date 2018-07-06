# encoding:utf-8
import sys, pygame, random
from pygame.locals import *

# -----------------------------参数-----------------------------
FPS = 30
WINDOWWIDTH = 640
WINDOWHEIGHT = 480
REVEALSPEED = 8
BOXSIZE = 40
GAPSIZE = 10
BOARDWIDTH = 10
BOARDHEIGHT = 7
assert (BOARDWIDTH*BOARDHEIGHT)%2 == 0, 'Board needs to have an even number of boxes for pairs of matches.'
XMARGIN = int((WINDOWWIDTH-(BOARDWIDTH*(BOXSIZE+GAPSIZE)))/2)
YMARGIN = int((WINDOWHEIGHT-(BOARDHEIGHT*(BOXSIZE+GAPSIZE)))/2)
# -------------------------------------------------------------

# -----------------------------颜色-----------------------------
#              R    G    B
GRAY        =(100, 100, 100)
NAVYBLUE    =( 60,  60, 100)
WHITE       =(255, 255, 255)
RED         =(255,   0,   0)
GREEN       =(  0, 255,   0)
BLUE        =(  0,   0, 255)
YELLOW      =(255, 255,   0)
ORANGE      =(255, 128,   0)
PURPLE      =(255,   0, 255)
CYAN        =(  0, 255, 255)

BGCOLOR = NAVYBLUE
LIGHTBGCOLOR = GRAY
BOXCOLOR = WHITE
HIGHLIGHTCOLOR = BLUE
# -------------------------------------------------------------

# ---------------------------定义变量---------------------------
DONUT = 'donut'
SQUARE = 'square'
DIAMOND = 'diamond'
LINES = 'lines'
OVAL = 'oval'
# ------------------------------------------------------------

ALLCOLORS = (RED, GREEN, BLUE, YELLOW, ORANGE, PURPLE, CYAN)
ALLSHAPES = (DONUT, SQUARE, DIAMOND, LINES, OVAL)
assert len(ALLCOLORS)*len(ALLSHAPES)*2 >= BOARDWIDTH*BOARDHEIGHT, \
    'Board is too big for the number of shapes/colors defined.'


def main():
    global FPSCLOCK, DISPLAYSURF
    # 1.如果在函数的开始处，有一条针对一个变量的全局语句，那么该变量为全局的。
    # 2.如果函数中一个变量的名称与一个全局变量的名称相同，并且该函数没有给该变量赋值，那么该变量是全局变量。
    # 3.如果函数中一个变量的名称与一个全局变量的名称相同，并且该函数给该变量赋了值，那么该变量是局部变量。
    # 4.如果没有一个全局变量与函数中的该变量具有相同的名称，那么该变量显然是一个局部变量。
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))

    mousex = 0  # 鼠标横坐标
    mousey = 0  # 鼠标纵坐标
    pygame.display.set_caption('Memory Game')

    mainBoard = getRandomizedBoard()
    revealedBoxes = generateRevealedBoxesData(False)

    firstSelection = None   # 储存按下的第一个Box的坐标

    DISPLAYSURF.fill(BGCOLOR)
    startGameAnimation(mainBoard)

    while True:     # main game loop
        mouseClicked = False

        DISPLAYSURF.fill(BGCOLOR)   # drawing the window
        drawBoard(mainBoard, revealedBoxes)

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYUP and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEMOTION:     # mouse moving
                mousex, mousey = event.pos
            elif event.type == MOUSEBUTTONUP:   # mouse click
                mousex, mousey = event.pos
                mouseClicked = True

        # 检查鼠标光标在哪一个box上
        boxx, boxy = getBoxAtPixel(mousex, mousey)
        if boxx != None and boxy != None:
            # 鼠标在box上
            if not revealedBoxes[boxx][boxy]:
                drawHighlightBox(boxx, boxy)
            if not revealedBoxes[boxx][boxy] and mouseClicked:    # 检查是否确实被按下
                revealBoxesAnimation(mainBoard, [(boxx, boxy)])
                revealedBoxes[boxx][boxy] = True  # set the box as revealed
                # 处理第一次点击方块
                if firstSelection == None:  # the current box is the first box clicked
                    firstSelection = (boxx, boxy)
                else:
                    icon1shape, icon1color = getShapeAndColor(mainBoard, firstSelection[0], firstSelection[1])
                    icon2shape, icon2color = getShapeAndColor(mainBoard, boxx, boxy)

                    # 处理不一致的一对图标
                    if icon1shape != icon2shape or icon1color != icon2color:
                        pygame.time.wait(1000)  # 1sec
                        coverBoxesAnimation(mainBoard, [(firstSelection[0], firstSelection[1]), (boxx, boxy)])
                        revealedBoxes[firstSelection[0]][firstSelection[1]] = False
                        revealedBoxes[boxx][boxy] = False
                    # 处理玩家胜利
                    elif hasWon(revealedBoxes):   # check is all pairs found
                        gameWonAnimation(mainBoard)
                        pygame.time.wait(2000)

                        # Reset the board
                        mainBoard = getRandomizedBoard()
                        revealedBoxes = generateRevealedBoxesData(False)

                        # Show the fully unrevealed board for a second
                        drawBoard(mainBoard, revealedBoxes)
                        pygame.display.update()
                        pygame.time.wait(1000)

                        # Replay the start game animation
                        startGameAnimation(mainBoard)
                    firstSelection = None   # reset firstSelection variable

        # Redraw the screen and wait a clock tick
        pygame.display.update()
        FPSCLOCK.tick(FPS)


# 创建“揭开的方块”数据结构
def generateRevealedBoxesData(val):
    revealedBoxes = []
    for i in range(BOARDWIDTH):
        revealedBoxes.append([val]*BOARDHEIGHT)
    return revealedBoxes


# 获取所有可能的图标
def getRandomizedBoard():
    icons = []
    for color in ALLCOLORS:
        for shape in ALLSHAPES:
            icons.append((shape, color))

    # 打乱并截取所有图标的列表
    random.shuffle(icons)
    numIconUsed = int(BOARDWIDTH*BOARDHEIGHT/2)     # 计算需要多少icon
    icons = icons[:numIconUsed]*2   # 每次生成一对
    random.shuffle(icons)

    # 将图标放置到游戏板上
    board = []
    for x in range(BOARDWIDTH):
        column = []
        for y in range(BOARDHEIGHT):
            column.append(icons[0])
            del icons[0]    # remove the icons as we assign them
        board.append(column)
    return board


# 将一个列表分割为列表的列表
def splitIntoGroupsOf(groupSize, theList):
    result = []
    for i in range(0, len(theList), groupSize):
        result.append(theList)
    return result


# 确定坐标系
def leftTopCoordsOfBox(boxx, boxy):
    left = boxx*(BOXSIZE+GAPSIZE)+XMARGIN
    top = boxy*(BOXSIZE+GAPSIZE)+YMARGIN
    return (left, top)


# 从像素坐标转化为方块坐标
def getBoxAtPixel(x, y):
    for boxx in range(BOARDWIDTH):
        for boxy in range(BOARDHEIGHT):
            left, top = leftTopCoordsOfBox(boxx, boxy)
            boxRect = pygame.Rect(left, top, BOXSIZE, BOXSIZE)
            if boxRect.collidepoint(x, y):
                return (boxx, boxy)
    return (None, None)


# 绘制图标及语法糖
def drawIcon(shape, color, boxx, boxy):
    quarter = int (BOXSIZE*0.25)    # syntactic sugar
    half =    int (BOXSIZE*0.5)     # syntactic sugar

    left, top = leftTopCoordsOfBox(boxx, boxy)  # get pixel coords from board coords
    # draw the shapes
    if shape == DONUT:
        pygame.draw.circle(DISPLAYSURF, color, (left+half, top+half), half-5)
        pygame.draw.circle(DISPLAYSURF, BGCOLOR, (left+half, top+half), quarter-5)
    elif shape == SQUARE:
        pygame.draw.rect(DISPLAYSURF, color, (left+quarter, top+quarter, BOXSIZE-half, BOXSIZE-half))
    elif shape == DIAMOND:
        pygame.draw.polygon(DISPLAYSURF,color, ((left+half, top), (left+BOXSIZE-1, top+half), (left+half, top+BOXSIZE-1), (left, top+half)))
    elif shape == LINES:
        for i in range(0, BOXSIZE, 4):
            pygame.draw.line(DISPLAYSURF, color, (left, top+i), (left+i, top))
            pygame.draw.line(DISPLAYSURF, color, (left+i, top+BOXSIZE-1), (left+BOXSIZE-1, top+i))
    elif shape == OVAL:
        pygame.draw.ellipse(DISPLAYSURF,color,(left, top+quarter, BOXSIZE, half))


# 获取游戏板控件的图标的形状和颜色的语法糖
def getShapeAndColor(board, boxx, boxy):
    # shape value for x,y spot is stored in board[x][y][0]
    # color value for x,y spot is stored in board[x][y][1]
    return board[boxx][boxy][0], board[boxx][boxy][1]


# 绘制盖住的方块
def drawBoxCovers(board, boxes, coverage):
    # draw boxex being covered/revealed."boxes" is a list of two-item lists,which have the x&y spot of the box
    for box in boxes:
        left, top = leftTopCoordsOfBox(box[0], box[1])
        pygame.draw.rect(DISPLAYSURF, BGCOLOR, (left, top, BOXSIZE, BOXSIZE))
        shape, color = getShapeAndColor(board, box[0], box[1])
        drawIcon(shape, color, box[0], box[1])
        if coverage > 0:    # only draw the cover if there is a coverage
            pygame.draw.rect(DISPLAYSURF, BOXCOLOR, (left, top, coverage, BOXSIZE))
    pygame.display.update()
    FPSCLOCK.tick(FPS)


# 揭开动画
def revealBoxesAnimation(board, boxesToReveal):
    for coverage in range(BOXSIZE, -1, -REVEALSPEED):
        drawBoxCovers(board,boxesToReveal, coverage)


# 覆盖动画
def coverBoxesAnimation(board, boxesToCover):
    for coverage in range(0, BOXSIZE+REVEALSPEED, REVEALSPEED):
        drawBoxCovers(board, boxesToCover, coverage)


# 绘制整个游戏板
def drawBoard(board, revealed):
    # draw all boxes
    for boxx in range(BOARDWIDTH):
        for boxy in range(BOARDHEIGHT):
            left, top = leftTopCoordsOfBox(boxx, boxy)
            if not revealed[boxx][boxy]:
                # draw a covered box
                pygame.draw.rect(DISPLAYSURF, BOXCOLOR, (left, top, BOXSIZE, BOXSIZE))
            else:
                # draw revealed icon
                shape, color = getShapeAndColor(board, boxx, boxy)
                drawIcon(shape, color, boxx, boxy)


# 绘制高亮边框
def drawHighlightBox(boxx, boxy):
    left, top = leftTopCoordsOfBox(boxx, boxy)
    pygame.draw.rect(DISPLAYSURF, HIGHLIGHTCOLOR, (left-5, top-5, BOXSIZE+10, BOXSIZE+10), 4)


# “开始游戏”动画
def startGameAnimation(board):
    # random reveal the boxes 8 at a time
    coveredBoxes = generateRevealedBoxesData(False)
    boxes = []
    for x in range(BOARDWIDTH):
        for y in range(BOARDHEIGHT):
            boxes.append((x, y))
    random.shuffle(boxes)
    boxGroups = splitIntoGroupsOf(8, boxes)
    # 揭开和盖住成组的方块
    drawBoard(board, coveredBoxes)
    for boxGroup in boxGroups:
        revealBoxesAnimation(board, boxGroup)
        coverBoxesAnimation(board, boxGroup)


# “游戏获胜”动画
def gameWonAnimation(board):
    # flash the background color when the player has won
    coveredBoxes = generateRevealedBoxesData(True)
    color1 = LIGHTBGCOLOR
    color2 = BGCOLOR

    for i in range(13):
        color1, color2 = color2, color1     # swap colors
        DISPLAYSURF.fill(color1)
        drawBoard(board, coveredBoxes)
        pygame.display.update()
        pygame.time.wait(300)


# 判断玩家是否已经获胜
def hasWon(revealedBoxes):
    # return True if all the boxes have been revealed, otherwise False
    for i in revealedBoxes:
        if False in i:
            return False
    return True


if __name__ == '__main__':
    main()
