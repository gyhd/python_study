import requests
import json
import collections


class getCovidData(object):

    def __init__(self):


        # 全国疫情数据
        self.chinatotal_url = 'https://view.inews.qq.com/g2/getOnsInfo?name=disease_other'
        self.chinatotal1_url = "https://view.inews.qq.com/g2/getOnsInfo?name=disease_h5"
        # 省区信息请求网址
        self.province_city_url = "https://view.inews.qq.com/g2/getOnsInfo?name=wuwei_ww_city_list_order"
        # 省一级及区一级详细信息请求地址(后面跟省/地级市中文)
        self.provincetotal_url = 'https://api.inews.qq.com/newsqa/v1/query/pubished/daily/list?province='
        # 国外疫情数据
        self.foreigntotal_url = 'https://view.inews.qq.com/g2/getOnsInfo?name=disease_foreign'

    def getProvinceCity(self):
        response = requests.get(self.province_city_url).json()  # 发出请求并json化处理
        data = json.loads(response['data'])  # 提取数据部分
        province = list()
        city = collections.defaultdict(list)
        for i in range(len(data)):
            province.append(data[i]['province'])
            city[data[i]['province']].append(data[i]['city'])
        return province, city

    def getChinaTotalData(self):
        response = requests.get(self.chinatotal_url).json()  # 发出请求并json化处理
        # 将获取到的json格式的字符串类型数据转换为python支持的字典类型数据
        data = json.loads(response['data'])
        # 所有的疫情数据,data['data']数据还是str的json格式需要转换为字典格式，包括：中国累积数据、各国数据(中国里面包含各省及地级市详细数据)、中国每日累积数据(1月13日开始)
        return data

    def getProvinceTotalData(self):
        response = requests.get(self.chinatotal1_url).json()
        data = json.loads(response['data'])
        areaTree = data['areaTree'][0]['children']
        return areaTree

    def getForeignTotalData(self):
        response = requests.get(self.foreigntotal_url).json()
        data = json.loads(response['data'])
        foreignList = data['foreignList']
        return foreignList





