file_name = input('Plz input file name:')
try:
    f = open(file_name)
    print('File:')
    for each_line in f:
        print(each_line, end='')
    f.close()
except OSError as reason:       # as reason 可加
    print('THERE IS A PROBLEM IN FILE OPERATION:' + str(reason))
except(OSError, TypeError):     # 两个异常一并
    print('ERROR')
finally:                        # 无论如何都会执行的代码
    f.close()
