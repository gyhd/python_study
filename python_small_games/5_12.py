
import plotly
import plotly.offline as py
import numpy as np
import plotly.graph_objs as go

# 设置offline
plotly.offline.init_notebook_mode(connected=True)

# 给出散点
N = 100
random_x = np.linspace(0, 1, N)
random_y0 = np.random.randn(N) + 5
random_y1 = np.random.randn(N)
random_y2 = np.random.randn(N) - 5

# 画散点图
trace0 = go.Scatter(x=random_x, y=random_y0,
                    mode='markers', name='malers')
data = [trace0]
py.plot(data)

# 画折线图
trace1 = go.Scatter(x=random_x, y=random_y2,
                     mode='lines')
py.plot([trace1])

# 给出折线图和散点图在同一幅图中出现的trace2:
trace2 = go.Scatter(x=random_x, y=random_y1,
                    mode='lines+markers', name='lines+markers')
data = [trace2]
py.plot(data)

# 把三幅图放在一起
data = [trace0, trace2, trace1]
py.plot(data)



