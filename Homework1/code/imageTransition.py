# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 2018
@author: xwy
@environment: python2.7
@dependency: opencv, numpy
"""

import numpy as np
import cv2 as cv
import math

noble = cv.imread('noble.jpg', cv.IMREAD_COLOR)
lena = cv.imread('lena.jpg', cv.IMREAD_COLOR)
result = cv.imread('noble.jpg', cv.IMREAD_COLOR)

if noble.shape[0] != lena.shape[0] or noble.shape[1] != lena.shape[1]:
  print("Different Size")
else:
  x_max = noble.shape[0]
  y_max = noble.shape[1]
  r_max = math.sqrt((x_max/2) * (x_max/2) + (y_max/2) * (y_max/2))
  t_max = 25
  for t in range(0, t_max + 1):
    r_t = math.floor(r_max * t / t_max)
    for x in range(0, x_max):
      for y in range(0, y_max):
        r = math.sqrt((x_max/2 - x) * (x_max/2 - x) + (y_max/2 - y) * (y_max/2 - y))
        if r < r_t:
          result.itemset((x, y, 0), lena.item(x, y, 0))
          result.itemset((x, y, 1), lena.item(x, y, 1))
          result.itemset((x, y, 2), lena.item(x, y, 2))
    cv.imshow('transition', result)
    cv.waitKey(1)
  cv.waitKey(0)
  cv.destroyAllWindows()