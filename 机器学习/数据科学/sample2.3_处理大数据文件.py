import pandas as pd

iris_filename = 'iris.csv'
# 采用区块分割
iris_chunks = pd.read_csv(iris_filename, names=['C1', 'C2', 'C3', 'C4', 'C5'], chunksize=10)
for chunk in iris_chunks:
    print(chunk.shape)
    print(chunk)
# 采用迭代器     可以动态的指定每一个pandas数据块的大小 
iris_iterator = pd.read_csv(iris_filename, names=['C1', 'C2', 'C3', 'C4', 'C5'], iterator=True)
print(iris_iterator.get_chunk(2))
print(iris_iterator.get_chunk(3))