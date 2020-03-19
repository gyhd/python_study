# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:12:33 2019

@author: Maibenben
"""
# 爬取李东风PDF文档,网址：http://www.math.pku.edu.cn/teachers/lidf/docs/textrick/index.htm


import urllib.request
import re
import os

# open the url and read
def getHtml(url):
    page = urllib.request.urlopen(url)
    html = page.read()
    page.close()
    return html

# compile the regular expressions and find
# all stuff we need
def getUrl(html):
    reg = r'(?:href|HREF)="?((?:http://)?.+?\.pdf)'
    url_re = re.compile(reg)
    url_lst = url_re.findall(html.decode('gb2312'))
    return(url_lst)

def getFile(url):
    file_name = url.split('/')[-1]
    u = urllib.request.urlopen(url)
    f = open(file_name, 'wb')

    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        f.write(buffer)
    f.close()
    print ("Sucessful to download" + " " + file_name)


root_url = 'https://wenku.baidu.com/view/'

raw_url = 'https://wenku.baidu.com/view/b61b11d6a6c30c2259019edb.htm'

html = getHtml(raw_url)
url_lst = getUrl(html)

os.mkdir('download')
os.chdir(os.path.join(os.getcwd(), 'ldf_download'))

for url in url_lst[:]:
    url = root_url + url
    getFile(url)





