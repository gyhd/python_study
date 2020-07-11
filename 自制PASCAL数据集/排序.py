
# -*- coding:utf8 -*-

import os


class BatchRename():
    '''
    批量重命名文件夹中的图片文件
    '''

    def __init__(self):
        # 我的图片文件夹路径
        self.path = r'C:\Users\gyhd\Desktop\laji\可回收物_鼠标'

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 224       # 设置第一个文件名
        n = 3       # 设置文件名长度，如000001，长度为6
        for item in filelist:
            # 这里修改的是jpg文件，如果要修改其他类型的文件，请手动将下面两个'.jpg'修改为对应的文件后缀
            if item.endswith('.jpeg'):
                n = 6 - len(str(i))
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), str(0) * n + str(i) + '.jpeg')
                try:
                    os.rename(src, dst)
                    print('converting %s to %s ...' % (src, dst))
                    i = i + 1

                except:
                    continue
        print('total %d to rename & converted %d jpegs' % (total_num, i))


if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()



