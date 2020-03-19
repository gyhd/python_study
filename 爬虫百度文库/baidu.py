# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 21:05:09 2019

@author: Maibenben
"""

from selenium import webdriver
from bs4 import BeautifulSoup

options = webdriver.ChromeOptions()
options.add_argument('user-agent="Mozilla/5.0 (Linux; Android 4.0.4; Galaxy Nexus Build/IMM76B) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.133 Mobile Safari/535.19"')
driver = webdriver.Chrome(chrome_options=options)
driver.get('https://wenku.baidu.com/view/aa31a84bcf84b9d528ea7a2c.html')

html = driver.page_source
bf1 = BeautifulSoup(html, 'lxml')
result = bf1.find_all(class_='rtcspage')

for each_result in result:
    bf2 = BeautifulSoup(str(each_result), 'lxml')
    texts = bf2.find_all('p')
    for each_text in texts:
        main_body = BeautifulSoup(str(each_text), 'lxml')
        for each in main_body.find_all(True):
            if each.name == 'span':
                print(each.string.replace('\xa0',''),end='')
                
            elif each.name == 'br':
                print('')



