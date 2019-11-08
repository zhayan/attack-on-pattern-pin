import math
import sys

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go


def toVector(start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dis = math.sqrt(dx**2 + dy ** 2)
    if dx == 0:
        return [[0,1,dy]]
    elif dx > 0:
        k = dy/abs(dx)
        if k > math.tan(math.pi*77/180):
            return[[0,1,dis]]
        elif k > math.tan(math.pi*55/180):
            return[[1,2,dis]]
        elif k > math.tan(math.pi*35/180):
            return[[1,1,dis]]
        elif k > math.tan(math.pi*13/180):
            return[[2,1,dis]]
        elif k > math.tan(math.pi*(-13)/180):
            return[[1,0,dis]]
        elif k > math.tan(math.pi*(-35)/180):
            return[[2,-1,dis]]
        elif k > math.tan(math.pi*(-55)/180):
            return[[1,-1,dis]]
        elif k > math.tan(math.pi*(-77)/180):
            return[[1,-2,dis]]
        else:
            return[[0,-1,dis]]
    else:
        k = dy/abs(dx)
        if k > math.tan(math.pi*77/180):
            return[[0,1,dis]]
        elif k > math.tan(math.pi*55/180):
            return[[-1,2,dis]]
        elif k > math.tan(math.pi*35/180):
            return[[-1,1,dis]]
        elif k > math.tan(math.pi*13/180):
            return[[-2,1,dis]]
        elif k > math.tan(math.pi*(-13)/180):
            return[[-1,0,dis]]
        elif k > math.tan(math.pi*(-35)/180):
            return[[-2,-1,dis]]
        elif k > math.tan(math.pi*(-55)/180):
            return[[-1,-1,dis]]
        elif k > math.tan(math.pi*(-77)/180):
            return[[-1,-2,dis]]
        else:
            return[[0,-1,dis]]

# Path of video
VIDEO_PATH = sys.argv[1]

VIDEO_NAME = str(VIDEO_PATH).split("/")[-1].split(".")[0]

TRACK_PATH = "../results/"+VIDEO_NAME+".csv"

data = pd.read_csv(TRACK_PATH)
xval = data["X"].values
yval = data["Y"].values

mindist = 0
points = []
points.append([xval[1],yval[1]])
#xval = xval[2:-7]
#yval = yval[2:-7]
for i in range(len(xval)):
    point = points.pop()
    if (xval[i] - point[0])*2 + (yval[i] - point[1])**2 > mindist:
        points.append(point)
        points.append([xval[i],yval[i]])
    else:
        points.append(point)


vectors = []
new_points = [points[0]]

for i in range(len(points) - 1):
    vector = toVector(points[i], points[i+1])
    vectors = vectors + vector
    
for vector in vectors:
    if abs(vector[2]) < 1.5:
        continue
    temp = new_points[-1]
    new_point = [temp[0] + vector[0]*vector[2],temp[1] + vector[1]*vector[2]]
    new_points.append(new_point)

mark = go.Scatter(x = [a[0] for a in new_points], y = [a[1] for a in new_points], mode = "markers")
trace = go.Scatter(x = [a[0] for a in new_points], y = [a[1] for a in new_points])
fig = go.Figure([trace, mark])
fig.update_yaxes(autorange="reversed")

#fig.show()

fig.write_image("../results/"+VIDEO_NAME+".jpeg")
