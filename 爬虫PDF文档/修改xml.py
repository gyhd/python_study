# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 14:50:56 2019

@author: Maibenben
"""


import os
import xml.etree.cElementTree as ET

def changesku(inputpath):
    listdir1 = os.listdir(inputpath)
    for file in listdir1:
        if file.endwith('xml'):
            file = os.path.join(inputpath,file)
            tree = ET.parse(file)
            root = tree.getroot()
            for object1 in root.findall('object'):
                for sku in object1.findall('name'):
                    if (sku.text == '005'):
                        sku.text == '008'
                        tree.write(file,encoding = 'utf-8')
                    else:
                        pass
        else:
            pass


if __name__ == '__main__':
    
    inputpath = r'C:\Users\Maibenben\Desktop\descript.xml'
    changesku(inputpath)



