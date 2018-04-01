# try:
#     f = open('data.txt', 'w')
#     for each_line in f:
#         print(each_line)
# except OSError as reason:
#     print('出错啦' + str(reason))
# finally:
#     f.close()   # ---->ERROR:not readable


try:
    with open('data.txt', 'w') as f:    # 省去close语句
        for each_line in f:
            print(each_line)
except OSError as reason:
    print('出错啦' + str(reason))
