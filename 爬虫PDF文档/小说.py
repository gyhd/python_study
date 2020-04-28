

import re
import urllib.request


def g():
    html = urllib.request.urlopen("http://www.quanshuwang.com/book/43/43362").read()
    # print(html)
    html = html.decode("gbk")
    req = '<li><a href="(.*?)" title=".*?">(.*?)</a></li>'
    urls = re.findall(req, html)

    for i in urls:
        novel_url = i[0]
        novel_name = i[1]
        chapt = urllib.request.urlopen(novel_url).read()
        chapt_html = chapt.decode("gbk")

        reg = '</script>&nbsp;&nbsp;&nbsp;&nbsp;(.*?)<script type="text/javascript">'

        reg = re.compile(reg, re.S)
        chapt_content = re.findall(reg, chapt_html)
        chapt_content = chapt_content[0].replace("&nbsp;&nbsp;&nbsp;&nbsp;", "")
        chapt_content = chapt_content.replace("<br />", "")

        print("正在下载：%s" % novel_name)
        with open('./img/{}.txt'.format(novel_name), 'w') as f:
            f.write(chapt_content)

g()







