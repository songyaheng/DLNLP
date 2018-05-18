# -*- coding:utf-8 -*-
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

img = cv2.imread("/Users/songyaheng/Downloads/tuzi.jpeg")

surf = cv2.xfeatures2d.SURF_create(10000)

key_query,desc_query = surf.detectAndCompute(img,None)

shape = np.shape(desc_query)
print(np.shape(desc_query))

fig = plt.figure()
ax = Axes3D(fig)
print(desc_query)
X = np.arange(0, shape[0], 1)
Y = np.arange(0, shape[1], 1)
X, Y = np.meshgrid(X, Y)
Z = desc_query[X, Y]
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

plt.show()

img = cv2.drawKeypoints(img,key_query,img)

cv2.imshow('sp',img)

cv2.waitKey(0)