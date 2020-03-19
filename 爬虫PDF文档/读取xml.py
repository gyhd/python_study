# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 14:29:30 2019

@author: Maibenben
"""


import xml.dom.minidom

#打开xml文档
dom = xml.dom.minidom.parse(r'C:\Users\Maibenben\Desktop\descript.xml')

#得到文档元素对象
root = dom.documentElement
print (root.nodeName)
print (root.nodeValue)
print (root.nodeType)
print (root.ELEMENT_NODE)


from xml.dom.minidom import parse
DOMTree=parse(r'C:\Users\Maibenben\Desktop\descript.xml')
#type(DOMTree)

booklist=DOMTree.documentElement

#返回xml的文档内容
booklist=DOMTree.documentElement
print (booklist.toxml())



