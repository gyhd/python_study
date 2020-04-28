
import matplotlib.pyplot as plt
import os

DATA_PATH = r'YOUR FILE PATH'
name_of_shower = 'per'
month_of_observation = '08'
showername = name_of_shower.upper()


def main():
    data = dict()
    file_list = sorted(os.listdir(DATA_PATH))
    for file in file_list:
        if not file.startswith('.'):
            if f'{showername}' in file:
                data.update(reader(file, print_flag=True))
    dates, observations, realdates = prepdata(data)
    plotter(dates, observations, realdates)


def reader(filename, print_flag=True):
    data = dict()
    counter = 0
    file_dir = DATA_PATH+filename
    with open(file_dir, 'r', encoding='utf8') as f_in:
        f_in.readline()
        for line in f_in:
            parts = line.strip('\n').split(';')
            data[parts[0]] = DataClass(parts)
            counter += 1
        if print_flag:
            print(f'In file: {filename}:\n{counter} observations were found.')
    return data


def prepdata(data):
    count = dict()
    day = int(1)
    hour = int(0)
    dates = list()
    while True:
        if day <= 31:
            if hour < 23:
                count[int(f'{month_of_observation}{day:0>2d}{hour:0>2d}')] = StatsClass()
                hour += 1
            if hour == 23:
                count[int(f'{month_of_observation}{day:0>2d}'f'{hour:0>2d}')] = StatsClass()
                hour = 0
                dates.append(f'{MonthClass(month_of_observation).monthname()}{day}')
                day += 1
        else:
            break

    for realid in data:
        if data[realid].month == month_of_observation:
            if data[realid].MMDDHH not in count:
                count[data[realid].MMDDHH] = StatsClass()
            count[data[realid].MMDDHH].sum += data[realid].observation
            count[data[realid].MMDDHH].count += 1

    group = list()
    numbers = list()
    for hour in sorted(count):
        group.append(hour)
        if count[hour].count == 0:
            numbers.append(int(0))
        else:
            numbers.append(count[hour].sum/count[hour].count)
    return group, numbers, dates


def plotter(dates, observations, realdates):
    fig = plt.figure()

    plt.bar(x=range(len(dates)), height=observations)
    plt.xticks(rotation='vertical')
    plt.xticks(range(0, len(dates), 24), realdates)
    month = MonthClass(month_of_observation).monthname()
    plt.title(f'Average number of Visual Meteors -{name_of_shower.upper()}- in {month}')

    plt.tight_layout()
    plt.show()
    fig.savefig(f'MeteorShowerPlotfor{showername}in{MonthClass(month_of_observation).monthname()}.pdf')


class DataClass:
    def __init__(self, parts):
        self.time = parts[3]
        self.observation = int(parts[12])
        self.month = self.time[6:8]
        self.day = self.time[9:11]
        self.hour = self.time[12:14]
        self.month_day = int(f'{self.month}{self.day}')
        self.MMDDHH = int(f'{self.month_day}{self.hour}')


class StatsClass:
    def __init__(self):
        self.count = 0
        self.sum = 0


class MonthClass:
    def __init__(self, month):
        self.month = month
        self.month_name = self.monthname()

    def monthname(self):
        name = 'Missing'
        if self.month == '01':
            name = 'Jan.'
        if self.month == '02':
            name = 'Feb.'
        if self.month == '03':
            name = 'Mar.'
        if self.month == '04':
            name = 'Apr.'
        if self.month == '05':
            name = 'May'
        if self.month == '06':
            name = 'Jun.'
        if self.month == '07':
            name = 'Jul.'
        if self.month == '08':
            name = 'Aug.'
        if self.month == '09':
            name = 'Sep.'
        if self.month == '10':
            name = 'Oct.'
        if self.month == '11':
            name = 'Nov.'
        if self.month == '12':
            name = 'Dec.'
        return name


main()
