
import os
import random

trainval_percent = 0.7      # 训练集和验证集 占 数据集 的比例
train_percent = 0.6         # 训练集 占 训练集和验证集 的比例
xmlfilepath = r'D:\data\new\VOCdevkit\VOC2007\Annotations'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

# 把路径修改为自己的Main文件夹路径
ftrainval = open(r'D:\data\new\VOCdevkit\VOC2007\ImageSets\Main\trainval.txt', 'w')
ftest = open(r'D:\data\new\VOCdevkit\VOC2007\ImageSets\Main\test.txt', 'w')
ftrain = open(r'D:\data\new\VOCdevkit\VOC2007\ImageSets\Main\train.txt', 'w')
fval = open(r'D:\data\new\VOCdevkit\VOC2007\ImageSets\Main\val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()



