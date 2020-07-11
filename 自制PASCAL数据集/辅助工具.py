
# coding=utf-8
import xml.dom.minidom
import os.path

i = 1       # 第一个文件命，如'000001.xml'
xmldir = r"C:\Users\gyhd\Desktop\savelabel"
imgsdir = r"C:\Users\gyhd\Desktop\laji\sum"


for xmlfile in os.listdir(xmldir):
    xmlname = os.path.splitext(xmlfile)[0]
    for pngfile in os.listdir(imgsdir):
        pngname = os.path.splitext(pngfile)[0]
        if pngname == xmlname:
            # 修改filename结点属性
            # 读取xml文件
            dom = xml.dom.minidom.parse(os.path.join(xmldir, xmlfile))
            root = dom.documentElement
            n = 6-len(str(i))
            # 获取标签对filename之间的值并赋予新值i
            root.getElementsByTagName('filename')[0].firstChild.data = str(0)*n + str(i) + '.jpg'       # 修改文件名
            root.getElementsByTagName('folder')[0].firstChild.data = 'VOC2007'                          # 修改文件夹名
            root.getElementsByTagName('path')[0].firstChild.data = '/home/guoweiwei/demo/darknet/data/voc/VOC2007/JPEGImages/' + str(0)*n + str(i) + '.jpg'     #修改图片路径名
            if root.getElementsByTagName('name')[0].firstChild.data == 'Remote control':                  # 修改标签名
                root.getElementsByTagName('name')[0].firstChild.data = 'Remote_control'

            # 将修改后的xml文件保存
            # xml文件修改前后的路径
            old_xmldir = os.path.join(xmldir, xmlfile)
            new_xmldir = os.path.join(xmldir, str(0)*n + str(i) + '.xml')
            # 打开并写入
            with open(old_xmldir, 'w') as fh:
                dom.writexml(fh)
            os.rename(old_xmldir, new_xmldir)
            i += 1
print('total number is ', i - 1)



