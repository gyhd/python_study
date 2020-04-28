from Covid19_data_get import getData


class ChinaData():

    def __init__(self):
        self.coviddata = getData.getCovidData()
        self.all_data = self.coviddata.getChinaTotalData()

    # 获取从1月13日起的一系列累计数据
    def china_total_data(self):
        chinaDayList = self.all_data["chinaDayList"]
        date_list1 = list()
        total_confirm = list()
        total_suspect = list()
        total_dead = list()
        total_heal = list()
        total_importedCase = list()
        for total in chinaDayList:
            date_list1.append(total['date'][:2] + "/" + total['date'][3:])
            total_confirm.append(int(total['confirm']))
            total_suspect.append(int(total['suspect']))
            total_dead.append(int(total['dead']))
            total_heal.append(int(total['heal']))
            total_importedCase.append(int(total['importedCase']))
        return date_list1, total_confirm, total_suspect, total_dead, total_heal, total_importedCase

    # 获取从1月20日起的一系列每日数据
    def china_everyday_data(self):
        chinaDayAddList = self.all_data["chinaDayAddList"]
        date_list2 = list()
        everyday_confirm = list()
        everyday_suspect = list()
        everyday_dead = list()
        everyday_heal = list()
        everyday_importedCase = list()
        for everyday in chinaDayAddList:
            date_list2.append(everyday['date'][:2] + "/" + everyday['date'][3:])
            everyday_confirm.append(int(everyday['confirm']))
            everyday_suspect.append(int(everyday['suspect']))
            everyday_dead.append(int(everyday['dead']))
            everyday_heal.append(int(everyday['heal']))
            everyday_importedCase.append(int(everyday['importedCase']))
        return date_list2, everyday_confirm, everyday_suspect, everyday_dead, everyday_heal, everyday_importedCase

    def main(self):
        self.china_total_data()
        self.china_everyday_data()


if __name__ == '__main__':
    world_data = ChinaData()
    world_data.main()
