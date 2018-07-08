import csv

iris_filename = 'iris.csv'
with open(iris_filename, 'rt') as data_stream:
    for n, row in enumerate(csv.DictReader(data_stream, dialect='excel')):
        if n == 0:
            print(row)
        else:
            break