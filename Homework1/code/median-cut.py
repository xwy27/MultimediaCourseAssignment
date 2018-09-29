# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 2018
@author: xwy
@environment: python2.7
@dependency: opencv, numpy
"""

from enum import Enum, unique
import numpy as np
import cv2 as cv
import math

class CV_RGBChannel(Enum):

  '''
  RBG Channel Value in OpenCV
  ATTENTION: BGR mode in OpenCV
  '''

  BLUE = 0
  GREEN = 1
  RED = 2

def takeR(elem):
  
  '''
  select red channel
  '''
  
  return elem[1][1][CV_RGBChannel.RED.value]


def takeG(elem):
  
  '''
  select green channel
  '''
  
  return elem[1][1][CV_RGBChannel.GREEN.value]


def takeB(elem):
  
  '''
  select blue channel
  '''
  
  return elem[1][1][CV_RGBChannel.BLUE.value]


def sortColorSpace(list, channel):
  
  '''
  sort the list by color channel values\n
  :param {list}list the list to be sorted\n
  :param {CV_RGBChannel/integer}channel the color channel used as the base to sort color space
  '''
  
  if channel == CV_RGBChannel.RED or\
    channel == CV_RGBChannel.RED.value:
    list.sort(key=takeR)
  elif channel == CV_RGBChannel.GREEN or\
    channel == CV_RGBChannel.GREEN.value:
    list.sort(key=takeG)
  elif channel == CV_RGBChannel.BLUE or\
    channel == CV_RGBChannel.BLUE.value:
    list.sort(key=takeB)


def splitColorSpace(list, channel):

  '''
  split sorted list by median value of color channel values\n
  :param {list}list  the color space to be split\n
  :param {CV_RGBChannel}channel  the color channel used to split color space
  '''

  if len(list) > 1:
    colorChannels = []
    for pixel in list:
      colorChannels.append(pixel[1][1][channel.value])
    
    half = len(colorChannels) // 2
    m = len(colorChannels) % 2
    median = 0
    if m > 0:
      median = colorChannels[half]
    else:
      median = (colorChannels[half] + colorChannels[half - 1]) / 2

    for pixel in list:
      if pixel[1][1][channel.value] >= median:
        pixel[1][2] = pixel[1][2] * 2 + 1
      else:
        pixel[1][2] = pixel[1][2] * 2 + 0



apple = cv.imread("redapple.jpg", cv.IMREAD_COLOR)
row, col, channel = apple.shape

# Initial color space
colorSpace = []
for i in range(0, row):
  for j in range(0, col):
    colorSpace.append([[i, j], apple[i, j], 0])

# Median-Cut Alogrithm
for i in range(0, 8):
  temp = []
  for j in range(0, int(math.pow(2, i))):
    temp2 = []
    for index, pixel in enumerate(colorSpace):
      if pixel[2] == j:
        temp2.append([index, pixel])
    if i % 3 == 0:
      sortColorSpace(temp2, CV_RGBChannel.RED)
      splitColorSpace(temp2, CV_RGBChannel.RED)
    elif i % 3 == 1:
      sortColorSpace(temp2, CV_RGBChannel.GREEN)
      splitColorSpace(temp2, CV_RGBChannel.GREEN)
    elif i % 3 == 2:
      sortColorSpace(temp2, CV_RGBChannel.BLUE)
      splitColorSpace(temp2, CV_RGBChannel.BLUE)

    temp = temp + temp2

  for t in temp:
    colorSpace[t[0]][2] = t[1][2]

# Calculate LUT
temp = []
for i in range(0, int(math.pow(2, 8))):
  sum = [0, 0, 0]
  total = 0
  for pixel in colorSpace:
    if pixel[2] == i:
      sum += pixel[1]
      total = total + 1
  if total > 0:
    sum =  [sum[0]/total, sum[1]/total, sum[2]/total]
    temp.append([i, sum])

# Map origin image RGB with LUT
for t in temp:
  for pixel in colorSpace:
    if pixel[2] == t[0]:
      pixel[1] = t[1]

for i in range(0, len(colorSpace)):
    apple[colorSpace[i][0][0], colorSpace[i][0][1]] = colorSpace[i][1]

cv.imshow('median-cut', apple)
cv.waitKey(0)
cv.destroyAllWindows()