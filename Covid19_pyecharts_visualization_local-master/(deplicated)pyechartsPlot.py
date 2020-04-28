from Covid19_data_get import getChinaData, getForeignData, getProvinceData
import pyecharts.options as opts
from pyecharts.charts import Line, Bar, Map

# 国内：日期，累计确诊，累计死亡
china_data = getChinaData.ChinaData()
date_list, china_confirm, _, china_dead, china_heal, china_importedCase = china_data.china_total_data()
# 国内各省 ： 省名，累计确诊数据
province_data = getProvinceData.ProvinceData()
province_name, province_total_confirm = province_data.province_total_data()
# 国外：国家，累计确诊，累计死亡
foreign_data = getForeignData.ForeignData()
foregin_name, foregin_total_confirm, _, foregin_dead, _ = foreign_data.foreign_total_data()


chinaMap = (
    Map()
        .add("累计确诊", [list(z) for z in zip(province_name, province_total_confirm)], maptype="china")
        # .add("现存确诊", [list(z) for z in zip(province_name, province_total_confirm)], maptype="china")
        # .add("累计确诊", [list(z) for z in zip(province_name, province_total_confirm)], maptype="world")
        .set_global_opts(
        title_opts=opts.TitleOpts(title="中国累计确诊数据"),

        visualmap_opts=opts.VisualMapOpts(is_piecewise=True, pieces=[{"max": 0, "label": '0人'}, {"min": 1, "max": 9, "label": '1-9人'}, {"min": 10, "max": 99, "label": '10-99人'},
                                           {"min": 100, "max": 499, "label": '100-499人'}, {"min": 500, "max": 999, "label": '500-999人'},
                                           {"min": 1000, "max": 9999, "label": '1000-9999人'},
                                           {"min": 10000, "label": '10000人及以上'}]),
    )
)
chinaLine = (
    Line()
        .add_xaxis(date_list)
        .add_yaxis('确诊', china_confirm, is_symbol_show=True, label_opts=opts.LabelOpts(is_show=False),
                   markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="max"), ]))
        .add_yaxis('死亡', china_dead, is_symbol_show=True, label_opts=opts.LabelOpts(is_show=False),
                   markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="max")]))
        .add_yaxis('治愈', china_heal, is_symbol_show=True, label_opts=opts.LabelOpts(is_show=False),
                   markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="max")]))
        .add_yaxis('输入病例', china_importedCase, is_symbol_show=True, label_opts=opts.LabelOpts(is_show=False),
                   markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="max")]))
        .set_global_opts(title_opts=opts.TitleOpts(title="国内疫情走势", subtitle="数据来源：腾讯新闻"),
                         yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=10, interval=3)))

)
foreignBar = (
    Bar()
        .add_xaxis(foregin_name[:10])
        .add_yaxis('确诊数', foregin_total_confirm[:10])
        .add_yaxis('死亡数', foregin_dead[:10])
        .set_global_opts(title_opts=opts.TitleOpts(title="国外疫情确诊Top10", subtitle="数据来源：腾讯新闻"))
)

chinaMap.render('chinamap.html')
chinaLine.render('chinaLine.html')
foreignBar.render('foreignBar.html')
