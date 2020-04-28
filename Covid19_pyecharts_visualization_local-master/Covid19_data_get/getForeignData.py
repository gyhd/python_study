from Covid19_data_get import getData


class ForeignData():

    def __init__(self):
        self.coviddata = getData.getCovidData()
        self.all_data = self.coviddata.getForeignTotalData()

    def foreign_total_data(self):
        foreign_data = self.all_data
        foreign_name = list()
        foreign_total_confirm = list()
        foreign_total_nowConfirm = list()
        foreign_total_dead = list()
        foreign_total_heal = list()

        for country in foreign_data:
            foreign_name.append(country['name'])
            foreign_total_confirm.append(country['confirm'])
            foreign_total_nowConfirm.append(country['nowConfirm'])
            foreign_total_dead.append(country['dead'])
            foreign_total_heal.append(country['heal'])

        return foreign_name, foreign_total_confirm, foreign_total_nowConfirm, foreign_total_dead, \
               foreign_total_heal
