# xml2csv.py
"""
需要train和test两个文件夹的 图片 和打标的 xml
"""
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

# os.chdir('C:/Dragunov/tensorflow/object_detection_API/文件准备/images/train')
# path = 'C:/Dragunov/tensorflow/object_detection_API/文件准备/images/train'
os.chdir('C:/Dragunov/tensorflow/object_detection_API/文件准备/images/test')
path = 'C:/Dragunov/tensorflow/object_detection_API/文件准备/images/test'


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = path
    xml_df = xml_to_csv(image_path)
    # xml_df.to_csv('train.csv', index=None)
    xml_df.to_csv('test.csv', index=None)
    print('Successfully converted xml to csv.')


main()

