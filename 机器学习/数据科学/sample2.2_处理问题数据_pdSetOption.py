import pandas as pd

pd.set_option('display.width', 100)         # 调整总列长
pd.set_option('display.max_colwidth', 100)  # 调整列宽
pd.set_option('display.max_columns', 10)    # 最大列数
pd.set_option('precision', 0)               # 小数点后面位数
pd.set_option('colheader_justify', 'left')  # 对齐方式：left  默认为右对齐

fake_dataset = pd.read_csv('sample2.2.csv', parse_dates=[0], error_bad_lines=False)     # parse_dates=[0] 第一列使用日期格式
                                                                                        # error_bad_lines=False忽略错误数据
print(fake_dataset)

# 使用fillna()替换缺失数据
print(fake_dataset.fillna(-1))  # 以-1替换
print(fake_dataset.fillna(fake_dataset.mean(axis=0)))