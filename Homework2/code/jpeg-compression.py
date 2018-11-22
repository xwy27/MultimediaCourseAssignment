# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 2018
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

class CV_YCbCrChannel(Enum):

  '''
  YCbCr Channel Value in OpenCV
  ATTENTION: YCrCb mode in OpenCV
  '''

  Y = 0
  Cr = 1
  Cb = 2

def createMatrix(m,n):
  '''
  Create zero m*n matrix
  '''
  matrix = [[0 for row in range(n)] for col in range(m)]
  return matrix

def toYCbCrSpace(src_img):
  '''
  Transform src_img into YCbCr color space image.
  src_img is in BGR mode and returns in YCrCb mode,
  which keeps the same in opencv
  '''
  row = src_img.shape[0]
  col = src_img.shape[1]

  for x in range(0, row):
    for y in range(0, col):
      r = src_img.item(x, y, CV_RGBChannel.RED.value)
      g = src_img.item(x, y, CV_RGBChannel.GREEN.value)
      b = src_img.item(x, y, CV_RGBChannel.BLUE.value)
      src_img.itemset((x, y, CV_YCbCrChannel.Y.value), 0.299*r+0.587*g+0.114*b)
      src_img.itemset((x, y, CV_YCbCrChannel.Cb.value), -0.168736*r-0.331264*g+0.5*b + 0.5 + 128)
      src_img.itemset((x, y, CV_YCbCrChannel.Cr.value), 0.5*r-0.418688*g-0.081312*b + 0.5 + 128)

  return src_img

def chromaSample(img):
  '''
  Sample chroma for image with 4:2:0 schema and
  return the result sample image
  '''
  row = img.shape[0]
  col = img.shape[1]
  x = 0
  y = 0
  while (x < row - 1):
    while (y < col - 1):
      sample_cb = img.item(x, y, CV_YCbCrChannel.Cb.value)
      sample_cr = img.item(x+1, y, CV_YCbCrChannel.Cr.value)
      for s_x in range(0, 2):
        for s_y in range(0, 2):
          # print('origin Cb: ', img.item(x+s_x, y+s_y, CV_YCbCrChannel.Cb.value), 'sample Cb: ', sample_cb)
          # print('origin Cr: ', img.item(x+s_x, y+s_y, CV_YCbCrChannel.Cr.value), 'sample Cr: ', sample_cr)
          img.itemset((x+s_x, y+s_y, CV_YCbCrChannel.Cb.value), sample_cb)
          img.itemset((x+s_x, y+s_y, CV_YCbCrChannel.Cr.value), sample_cr)
      y += 2
    x += 2
  return img

def DCT_1D_COL(block):
  '''
  Calculate G(i,v)
  '''
  S = createMatrix(8, 8)
  for i in range(0, 8):
    for v in range(0, 8):
      temp = 0
      for j in range(0, 8):
        temp += math.cos((2*j+1)*v*math.pi/16)*block[i][j]
      temp = temp * (math.sqrt(2)/2 if (v==0) else 1) / 2
      S[i][v] = temp
  return S 

def DCT_1D_ROW(block):
  '''
  Calculate F(u,v)
  '''
  S = createMatrix(8, 8)
  for u in range(0, 8):
    for v in range(0, 8):
      temp = 0
      for i in range(0, 8):
        temp += math.cos((2*i+1)*u*math.pi/16)*block[i][v]
      temp = temp * (math.sqrt(2)/2 if (u==0) else 1) / 2
      S[u][v] = round(temp)
  return S

# def IDCT_1D_COL(block):
# def IDCT_1D_ROW(block):

img = cv.imread('../img/cartoon.jpg', cv.IMREAD_COLOR)
img = cv.resize(img,(1000,720),interpolation = cv.INTER_CUBIC)

# Libaray YCbCr
img_ycbcr = cv.cvtColor(img,cv.COLOR_BGR2YCrCb)
# transform rgb to YCbCr
img_ycrcb = toYCbCrSpace(img)
# Sample image
img_sample = chromaSample(img_ycrcb)

cv.imshow('origin', img)
cv.imshow('resize', res)
# cv.imshow('ycbcr', img_ycbcr)
cv.waitKey(0)
cv.destroyAllWindows()