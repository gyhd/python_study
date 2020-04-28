from Covid19_data_get import getData
import json


class ProvinceData():

    def __init__(self):
        self.coviddata = getData.getCovidData()
        self.all_data = self.coviddata.getProvinceTotalData()

    def province_total_data(self):
        areaTree = self.all_data
        province_name = list()
        province_total_confirm = list()
        province_total_suspect = list()
        province_total_dead = list()
        province_total_heal = list()
        # province_total_importedCase = list()
        for province in areaTree:
            province_name.append(province['name'])
            province_total_confirm.append(province['total']['confirm'])
            province_total_suspect.append(province['total']['suspect'])
            province_total_dead.append(province['total']['dead'])
            province_total_heal.append(province['total']['heal'])
            # province_total_importedCase.append(province['total']['importedCase'])
        # 将省份名称和确诊人数对应打包为字典，用于ECharts地图可视化
        province_total_confirm_dict = {'name': province_name, 'value': province_total_confirm}
        with open('province_total.json', 'w', encoding='utf-8') as f:
            json.dump(province_total_confirm_dict, f, ensure_ascii=False)
        return province_name, province_total_confirm


    def province_today_data(self):
        """获取各省今日数据"""
        areaTree = self.all_data['areaTree'][0]['children']
        province_name = list()
        province_today_confirm = list()
        province_today_importedCase = list()
        province_today_dead = list()
        province_today_heal = list()
        for province in areaTree:
            province_name.append(province['name'])
            province_today_confirm.append(province['today']['confirm'])
            province_today_importedCase.append(province['today']['importedCase'])
            province_today_dead.append(province['total']['dead'])
            province_today_heal.append(province['total']['heal'])
        return province_name, province_today_confirm, province_today_importedCase, province_today_dead, \
               province_today_heal

    def main(self):
        self.province_total_data()


if __name__ == '__main__':
    world_data = ProvinceData()
    world_data.main()
