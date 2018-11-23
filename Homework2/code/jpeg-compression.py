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
import copy

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

class QUANTIFYMATRIX(Enum):

  '''
  Quantify Matrix Enumerator
  '''

  LUMINANCE = True
  CHROMA = False

LUMINANCE_QUANTIFY_MATRIX = [
  [16,11,10,16,24,40,51,61],
  [12,12,14,19,26,58,60,55],
  [14,13,16,24,40,57,69,56],
  [14,17,22,29,51,87,80,62],
  [18,22,37,56,68,109,103,77],
  [24,35,55,64,81,104,113,92],
  [49,64,78,87,103,121,120,101],
  [72,92,95,98,112,100,103,99]
]

CHROMA_QUANTIFY_MATRIX = [
  [17,18,24,47,99,99,99,99],
  [18,21,26,66,99,99,99,99],
  [24,26,56,99,99,99,99,99],
  [47,66,99,99,99,99,99,99],
  [99,99,99,99,99,99,99,99],
  [99,99,99,99,99,99,99,99],
  [99,99,99,99,99,99,99,99],
  [99,99,99,99,99,99,99,99]
]


def createMatrix(m,n):
  '''
  Create zero m*n matrix
  '''
  matrix = [[0 for row in range(n)] for col in range(m)]
  return matrix

def RGBtoYCbCrSpace(img):
  '''
  Transform img from RGB to YCbCr color space image.
  img is in BGR mode and returns in YCrCb mode,
  which keeps the same in opencv
  '''
  row = img.shape[0]
  col = img.shape[1]

  for x in range(0, row):
    for y in range(0, col):
      r = img.item(x, y, CV_RGBChannel.RED.value)
      g = img.item(x, y, CV_RGBChannel.GREEN.value)
      b = img.item(x, y, CV_RGBChannel.BLUE.value)
      img.itemset((x, y, CV_YCbCrChannel.Y.value), 0.299*r+0.587*g+0.114*b)
      img.itemset((x, y, CV_YCbCrChannel.Cb.value), -0.1687*r-0.3313*g+0.5*b + 128)
      img.itemset((x, y, CV_YCbCrChannel.Cr.value), 0.5*r-0.418*g-0.0813*b + 128)

  return img

def YCbCrtoRGBSpace(img):
  '''
  Transform img from YCbCr to RGB color space image.
  img is in BGR mode and returns in YCrCb mode,
  which keeps the same in opencv
  '''
  row = img.shape[0]
  col = img.shape[1]

  for x in range(0, row):
    for y in range(0, col):
      Y = img.item(x, y, CV_YCbCrChannel.Y.value)
      Cb = img.item(x, y, CV_YCbCrChannel.Cb.value) - 128
      Cr = img.item(x, y, CV_YCbCrChannel.Cr.value) - 128

      r = Y+1.402*Cr if (Y+1.402*Cr > 0) else 0
      g = Y-0.34414*Cb-0.71414*Cr if (Y-0.34414*Cb-0.71414*Cr > 0) else 0
      b = Y+1.772*Cb if (Y+1.772*Cb > 0) else 0
      img.itemset((x, y, CV_RGBChannel.RED.value), r)
      img.itemset((x, y, CV_RGBChannel.GREEN.value), g)
      img.itemset((x, y, CV_RGBChannel.BLUE.value), b)

  return img

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
      sample_cb = img.item(x, y,
        CV_YCbCrChannel.Cb.value)
      sample_cr = img.item(x+1, y,
        CV_YCbCrChannel.Cr.value)
      for s_x in range(0, 2):
        for s_y in range(0, 2):
          img.itemset((x+s_x, y+s_y,
            CV_YCbCrChannel.Cb.value), sample_cb)
          img.itemset((x+s_x, y+s_y,
            CV_YCbCrChannel.Cr.value), sample_cr)
      y += 2
    x += 2
  return img

def DCT_1D_COL(block):
  '''
  Calculate G(i,v) from f(i, j)
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
  Calculate F(u,v) from G(i, v)
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

def IDCT_1D_ROW(matrix):
  '''
  Calculate G(i,v) from F(u,v)
  '''
  S = createMatrix(8,8)
  for i in range(0, 8):
    for v in range(0, 8):
      temp = 0
      for u in range(0, 8):
        c = math.sqrt(2)/2 if(u==0) else 1
        temp += c/2*math.cos((2*i+1)*u*math.pi/16)*matrix[u][v]
      S[i][v] = round(temp)
  return S

def IDCT_1D_COL(matrix):
  '''
  Calculate f(i,j) from G(i,v)
  '''
  S = createMatrix(8,8)
  for i in range(0, 8):
    for j in range(0, 8):
      temp = 0
      for v in range(0, 8):
        c = math.sqrt(2)/2 if(v==0) else 1
        temp += c/2*math.cos((2*j+1)*v*math.pi/16)*matrix[i][v]
      S[i][j] = round(temp) + 128
  return S

def DCTImg(img):
  '''
  Perform DCT on image
  '''
  row = img.shape[0]
  col = img.shape[1]
  x = 0
  y = 0
  while (x < row - 7):
    while (y < col - 7):
      y_matrix = createMatrix(8, 8)
      cb_matrix = createMatrix(8, 8)
      cr_matrix = createMatrix(8, 8)
      for i in range(0, 8):
        for j in range(0, 8):
          y_matrix[i][j] = img.item(x + i, y + j, CV_YCbCrChannel.Y.value)-128
          cb_matrix[i][j] = img.item(x + i, y + j, CV_YCbCrChannel.Cb.value)-128
          cr_matrix[i][j] = img.item(x + i, y + j, CV_YCbCrChannel.Cr.value)-128
      y_matrix = DCT_1D_COL(y_matrix)
      cb_matrix = DCT_1D_COL(cb_matrix)
      cr_matrix = DCT_1D_COL(cr_matrix)
      
      y_matrix = DCT_1D_ROW(y_matrix)
      cb_matrix = DCT_1D_ROW(cb_matrix)
      cr_matrix = DCT_1D_ROW(cr_matrix)

      for i in range(0, 8):
        for j in range(0, 8):
          img.itemset((x + i, y + j, CV_YCbCrChannel.Y.value), y_matrix[i][j])
          img.itemset((x + i, y + j, CV_YCbCrChannel.Cb.value), cb_matrix[i][j])
          img.itemset((x + i, y + j, CV_YCbCrChannel.Cr.value), cr_matrix[i][j])
      y += 8
    x += 8
  return img

def IDCTImg(img):
  '''
  Perform IDCT on image
  '''
  row = img.shape[0]
  col = img.shape[1]
  x = 0
  y = 0
  while (x < row - 7):
    while (y < col - 7):
      y_matrix = createMatrix(8, 8)
      cb_matrix = createMatrix(8, 8)
      cr_matrix = createMatrix(8, 8)
      for i in range(0, 8):
        for j in range(0, 8):
          y_matrix[i][j] = img.item(x + i, y + j, CV_YCbCrChannel.Y.value)
          cb_matrix[i][j] = img.item(x + i, y + j, CV_YCbCrChannel.Cb.value)
          cr_matrix[i][j] = img.item(x + i, y + j, CV_YCbCrChannel.Cr.value)
      y_matrix = IDCT_1D_ROW(y_matrix)
      cb_matrix = IDCT_1D_ROW(cb_matrix)
      cr_matrix = IDCT_1D_ROW(cr_matrix)
      
      y_matrix = IDCT_1D_COL(y_matrix)
      cb_matrix = IDCT_1D_COL(cb_matrix)
      cr_matrix = IDCT_1D_COL(cr_matrix)

      for i in range(0, 8):
        for j in range(0, 8):
          img.itemset((x + i, y + j, CV_YCbCrChannel.Y.value), y_matrix[i][j])
          img.itemset((x + i, y + j, CV_YCbCrChannel.Cb.value), cb_matrix[i][j])
          img.itemset((x + i, y + j, CV_YCbCrChannel.Cr.value), cr_matrix[i][j])
      y += 8
    x += 8
  return img

def Quantify(matrix, quantifyMatrix):
  '''
  Quantify matrix with given quantifyMatrix
  quantifyMatrix = true means Luminance Quantify
  quantifyMatrix = false means Chroma Quantify
  '''
  S = createMatrix(8, 8)
  if (quantifyMatrix == QUANTIFYMATRIX.LUMINANCE.value):
    for i in range(0, 8):
      for j in range(0, 8):
        temp = matrix[i][j] / LUMINANCE_QUANTIFY_MATRIX[i][j]
        S[i][j] = (int)(round(temp))
  else: # QUANTIFYMATRIX.CHROMA.value
    for i in range(0, 8):
      for j in range(0, 8):
        temp = matrix[i][j] / CHROMA_QUANTIFY_MATRIX[i][j]
        S[i][j] = (int)(round(temp))
  return S

def DeQuantify(matrix, quantifyMatrix):
  '''
  DeQuantify matrix with given quantifyMatrix
  quantifyMatrix = true means Luminance Quantify
  quantifyMatrix = false means Chroma Quantify
  '''
  S = createMatrix(8, 8)
  if (quantifyMatrix == QUANTIFYMATRIX.LUMINANCE.value):
    for i in range(0, 8):
      for j in range(0, 8):
        S[i][j] = matrix[i][j] * LUMINANCE_QUANTIFY_MATRIX[i][j]
  else: # QUANTIFYMATRIX.CHROMA.value
    for i in range(0, 8):
      for j in range(0, 8):
        S[i][j] = matrix[i][j] * CHROMA_QUANTIFY_MATRIX[i][j]
  return S

def QuantifyImg(img):
  '''
  Perform quantify on image
  '''
  row = img.shape[0]
  col = img.shape[1]
  x = 0
  y = 0
  while (x < row - 7):
    while (y < col - 7):
      y_matrix = createMatrix(8, 8)
      cb_matrix = createMatrix(8, 8)
      cr_matrix = createMatrix(8, 8)
      for i in range(0, 8):
        for j in range(0, 8):
          y_matrix[i][j] = img.item(x + i, y + j, CV_YCbCrChannel.Y.value)
          cb_matrix[i][j] = img.item(x + i, y + j, CV_YCbCrChannel.Cb.value)
          cr_matrix[i][j] = img.item(x + i, y + j, CV_YCbCrChannel.Cr.value)
      y_matrix = Quantify(y_matrix, QUANTIFYMATRIX.LUMINANCE)
      cb_matrix = Quantify(cb_matrix, QUANTIFYMATRIX.CHROMA)
      cr_matrix = Quantify(cr_matrix, QUANTIFYMATRIX.CHROMA)

      for i in range(0, 8):
        for j in range(0, 8):
          img.itemset((x + i, y + j, CV_YCbCrChannel.Y.value), y_matrix[i][j])
          img.itemset((x + i, y + j, CV_YCbCrChannel.Cb.value), cb_matrix[i][j])
          img.itemset((x + i, y + j, CV_YCbCrChannel.Cr.value), cr_matrix[i][j])
      y += 8
    x += 8
  return img

def DeQuantifyImg(img):
  '''
  Perform dequantify on image
  '''
  row = img.shape[0]
  col = img.shape[1]
  x = 0
  y = 0
  while (x < row - 7):
    while (y < col - 7):
      y_matrix = createMatrix(8, 8)
      cb_matrix = createMatrix(8, 8)
      cr_matrix = createMatrix(8, 8)
      for i in range(0, 8):
        for j in range(0, 8):
          y_matrix[i][j] = img.item(x + i, y + j, CV_YCbCrChannel.Y.value)
          cb_matrix[i][j] = img.item(x + i, y + j, CV_YCbCrChannel.Cb.value)
          cr_matrix[i][j] = img.item(x + i, y + j, CV_YCbCrChannel.Cr.value)
      y_matrix = DeQuantify(y_matrix, QUANTIFYMATRIX.LUMINANCE)
      cb_matrix = DeQuantify(cb_matrix, QUANTIFYMATRIX.CHROMA)
      cr_matrix = DeQuantify(cr_matrix, QUANTIFYMATRIX.CHROMA)

      for i in range(0, 8):
        for j in range(0, 8):
          img.itemset((x + i, y + j, CV_YCbCrChannel.Y.value), y_matrix[i][j])
          img.itemset((x + i, y + j, CV_YCbCrChannel.Cb.value), cb_matrix[i][j])
          img.itemset((x + i, y + j, CV_YCbCrChannel.Cr.value), cr_matrix[i][j])
      y += 8
    x += 8
  return img

def LossDegree(src, res):
  '''
  Calculate the distortion of RGB
  '''
  R_loss = 0
  G_loss = 0
  B_loss = 0
  R_snr = 0
  G_snr = 0
  B_snr = 0
  row = src.shape[0]
  col = src.shape[1]
  for i in range(0, row):
    for j in range(0, col):
      R_loss += math.pow(src.item(i, j, CV_RGBChannel.RED.value) - res.item(i, j, CV_RGBChannel.RED.value), 2)
      R_snr += math.pow(src.item(i, j, CV_RGBChannel.RED.value), 2)
      G_loss += math.pow(src.item(i, j, CV_RGBChannel.GREEN.value) - res.item(i, j, CV_RGBChannel.GREEN.value), 2)
      G_snr += math.pow(src.item(i, j, CV_RGBChannel.GREEN.value), 2)
      B_loss += math.pow(src.item(i, j, CV_RGBChannel.BLUE.value) - res.item(i, j, CV_RGBChannel.BLUE.value), 2)
      B_snr += math.pow(src.item(i, j, CV_RGBChannel.BLUE.value), 2)
      # print(src.item(i, j, CV_RGBChannel.BLUE.value), res.item(i, j, CV_RGBChannel.BLUE.value))
  R_snr = 10*math.log10(R_snr / R_loss)
  G_snr = 10*math.log10(G_snr / G_loss)
  B_snr = 10*math.log10(B_snr / B_loss)
  R_loss = round(R_loss / (row * col))
  G_loss = round(G_loss / (row * col))
  B_loss = round(B_loss / (row * col))
  return R_loss, R_snr, G_loss, G_snr, B_loss, B_snr

# Origin photo
img = cv.imread('../img/cartoon.jpg', cv.IMREAD_COLOR)
img = cv.resize(img,(1000,720),interpolation = cv.INTER_CUBIC)
img_c = copy.deepcopy(img)
# img_copy = cv.cvtColor(img_c, cv.COLOR_BGR2YCR_CB)

# Transform rgb to YCbCr
my_ycbcr = RGBtoYCbCrSpace(img)
# idct_img = RGBtoYCbCrSpace(img)

# Chroma sample image
sample_img = chromaSample(my_ycbcr)
# idct_img = chromaSample(my_ycbcr)

# DCT image
dct_img = DCTImg(sample_img)

# # Quantify image
quant_img = QuantifyImg(dct_img)

# # Dequantify image
dequant_img = DeQuantifyImg(quant_img)

# IDCT image
idct_img = IDCTImg(dequant_img)
# idct_img = IDCTImg(dct_img)

# # reconstruct image
reconstruct_img = YCbCrtoRGBSpace(idct_img)

a,A,b,B,c,C=LossDegree(img_c, reconstruct_img)
print 'cartoon.jpg'
print 'R Channel MSE: ', a, 'R Channel SNR: ', A
print 'G Channel MSE: ', b, 'G Channel SNR: ', B
print 'B Channel MSE: ', c, 'B Channel SNR: ', C

# cv.imshow('my_img', reconstruct_img)
# cv.imshow('origin_img', img_c)
# cv.waitKey(0)
# cv.destroyAllWindows()