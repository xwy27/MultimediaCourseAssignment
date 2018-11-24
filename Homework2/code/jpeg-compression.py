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

def ProcessLog(msg):
  print 15*'-' + msg + 15*'-' + '\n'

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

def Distortion(src, res, imgPath):
  '''
  Calculate the distortion of RGB
  '''
  R_mse = 0
  G_mse = 0
  B_mse = 0
  R_snr = 0
  G_snr = 0
  B_snr = 0
  row = src.shape[0]
  col = src.shape[1]
  for i in range(0, row):
    for j in range(0, col):
      R_mse += math.pow(src.item(i, j, CV_RGBChannel.RED.value) - res.item(i, j, CV_RGBChannel.RED.value), 2)
      R_snr += math.pow(src.item(i, j, CV_RGBChannel.RED.value), 2)
      G_mse += math.pow(src.item(i, j, CV_RGBChannel.GREEN.value) - res.item(i, j, CV_RGBChannel.GREEN.value), 2)
      G_snr += math.pow(src.item(i, j, CV_RGBChannel.GREEN.value), 2)
      B_mse += math.pow(src.item(i, j, CV_RGBChannel.BLUE.value) - res.item(i, j, CV_RGBChannel.BLUE.value), 2)
      B_snr += math.pow(src.item(i, j, CV_RGBChannel.BLUE.value), 2)
  R_snr = 10*math.log10(R_snr / R_mse)
  G_snr = 10*math.log10(G_snr / G_mse)
  B_snr = 10*math.log10(B_snr / B_mse)
  R_mse = round(R_mse / (row * col))
  G_mse = round(G_mse / (row * col))
  B_mse = round(B_mse / (row * col))

  print 15*'-' + 'Distortion For ' + imgPath + 15*'-'
  print 'R Channel MSE: ', R_mse, 'R Channel SNR: ', R_snr
  print 'G Channel MSE: ', G_mse, 'G Channel SNR: ', G_snr
  print 'B Channel MSE: ', B_mse, 'B Channel SNR: ', B_snr

def ZigZag(matrix):
  '''
  Z scan a 8*8 matrix and returns a z scan array
  '''
  ZigZag = []
  i = 0
  j = 0
  tag = 1 # 1:right, 2:right-down, 3:down, 4:left-up
  # (0, 0) to (7, 0)
  for t in range(0, 36):
    ZigZag.append(matrix[i][j])
    if (tag == 1):
      j += 1
      tag = 2
    elif (tag == 3):
      if (i < 7):
        i += 1
        tag = 4
      else:
        tag = 1
    elif (tag == 2):
      i += 1
      j -= 1
      tag = 3 if (j ==0) else 2
    elif (tag == 4):
      j += 1
      i -= 1
      tag = 1 if (i == 0) else 4
  
  j += 1
  tag = 4
  # (7, 1) to (7, 7)
  for t in range(36, 64):
    ZigZag.append(matrix[i][j])
    if (tag == 4):
      i -= 1
      j += 1
      tag = 3 if (j == 7) else 4
    elif (tag == 3):
      i += 1
      tag = 2
    elif (tag == 2):
      i += 1
      j -= 1
      tag = 1 if (i == 7) else 2
    elif (tag == 1):
      j += 1
      tag = 4

  return ZigZag

def DeZigZag(array):
  '''
  Z scan the 8 * 8 matrix and put item in array
  in order to recover the matrix from z scan array
  '''
  matrix = createMatrix(8, 8)
  i = 0
  j = 0
  tag = 1 # 1:right, 2:right-down, 3:down, 4:left-up
  # (0, 0) to (7, 0)
  for t in range(0, 36):
    matrix[i][j] = array[t]
    if (tag == 1):
      j += 1
      tag = 2
    elif (tag == 3):
      if (i < 7):
        i += 1
        tag = 4
      else:
        tag = 1
    elif (tag == 2):
      i += 1
      j -= 1
      tag = 3 if (j ==0) else 2
    elif (tag == 4):
      j += 1
      i -= 1
      tag = 1 if (i == 0) else 4
  
  j += 1
  tag = 4
  # (7, 1) to (7, 7)
  for t in range(36, 64):
    matrix[i][j] = array[t]
    if (tag == 4):
      i -= 1
      j += 1
      tag = 3 if (j == 7) else 4
    elif (tag == 3):
      i += 1
      tag = 2
    elif (tag == 2):
      i += 1
      j -= 1
      tag = 1 if (i == 7) else 2
    elif (tag == 1):
      j += 1
      tag = 4

  return matrix

def ZigZagImg(img):
  '''
  Z scan the image
  '''
  Y_zig = []
  Cb_zig = []
  Cr_zig = []
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
      Y_zig.append(ZigZag(y_matrix))
      Cb_zig.append(ZigZag(cb_matrix))
      Cr_zig.append(ZigZag(cr_matrix))
      y += 8
    x += 8
  return Y_zig, Cb_zig, Cr_zig

def DeZigZagImg(img, Y_zig, Cb_zig, Cr_zig):
  '''
  Recover img from Z scan
  '''
  row = img.shape[0]
  col = img.shape[1]
  x = 0
  y = 0
  while (x < row - 7):
    while (y < col - 7):
      for t in range(0, len(Y_zig)):
        y_matrix = DeZigZag(Y_zig[t])
        cb_matrix = DeZigZag(Cb_zig[t])
        cr_matrix = DeZigZag(Cr_zig[t])
        for i in range(0, 8):
          for j in range(0, 8):
            img.itemset((x + i, y + j, CV_YCbCrChannel.Y.value), y_matrix[i][j])
            img.itemset((x + i, y + j, CV_YCbCrChannel.Cb.value), cb_matrix[i][j])
            img.itemset((x + i, y + j, CV_YCbCrChannel.Cr.value), cr_matrix[i][j])
      y += 8
    x += 8
  return img

def RLC(array):
  '''
  Return the run-length encode of the array
  '''
  count = 0
  num = 0
  rlc = []
  for i in range(1, len(array)):
    if (array[i] != 0):
      rlc.append([count, array[i]])
      count = 0
    else:
      count += 1
  rlc.append([0, 0])
  return rlc

def DeRLC(array):
  '''
  Return the array encoded by rlc
  '''
  res = []
  for i in range(0, len(array)):
    if (array[i][1] == 0):
      for t in range(len(res), 63):
        res.append(0)
    else:
      for j in range(0, array[i][0]):
        res.append(0)
      res.append(array[i][1])
  return res

def RLC_Zig(zig):
  '''
  RLC AC signal in Zig
  '''
  zig_rlc = []
  for i in range(0, len(zig)):
    zig_rlc.append(RLC(zig[i]))
  return zig_rlc

def DeRLC_zig(zig_rlc):
  '''
  DeRLC AC signal in Zig
  '''
  dezig = []
  for i in range(0, len(zig_rlc)):
    dezig.append(DeRLC(zig_rlc[i]))
  return dezig

def DPCM(zig):
  '''
  DPCM DC signal in Zig
  '''
  dpcm = []
  d = zig[0][0]
  dpcm.append(d)
  for i in range(1, len(zig)):
    dpcm.append(zig[i][0]-d)
  return dpcm

def DeDPCM(dpcm):
  '''
  DeDPCM DC signal from Zig
  '''
  DC = []
  d = dpcm[0]
  DC.append(d)
  for i in range(1, len(dpcm)):
    DC.append(dpcm[i]+d)
  return DC

def Merge_RLC_DPCM(rlc, dpcm):
  '''
  Merge the RLC & DPCM into Zig
  '''
  zig = []
  for i in range(0, len(dpcm)):
    temp = []
    temp.append(dpcm[i])
    for j in range(0, len(rlc[i])):
      temp.append(rlc[i][j])
    zig.append(temp)
  return zig


ProcessLog(' Start Read Image ')
# Origin photo
imgPath = '../img/cartoon.jpg'
img = cv.imread(imgPath, cv.IMREAD_COLOR)
img = cv.resize(img,(1000,720),interpolation = cv.INTER_CUBIC)
img_c = copy.deepcopy(img)
# img_copy = cv.cvtColor(img_c, cv.COLOR_BGR2YCR_CB)
ProcessLog(' End Read Image ')


ProcessLog(' Start Transform Color Space ')
# Transform rgb to YCbCr
ycbcr_img = RGBtoYCbCrSpace(img)
ProcessLog(' End Transform Color Space ')


ProcessLog(' Start Chroma Sampling ')
# Chroma sample image
sample_img = chromaSample(ycbcr_img)
ProcessLog(' End Chroma Sampling ')


ProcessLog(' Start DCT ')
# DCT image
dct_img = DCTImg(sample_img)
ProcessLog(' End DCT ')


ProcessLog(' Start Quantization ')
# Quantify image
quant_img = QuantifyImg(dct_img)
ProcessLog(' End Quantization ')

ProcessLog(' Start ZigZag Scan ')
# Zigzag scan image
Y_zig, Cb_zig, Cr_zig = ZigZagImg(quant_img)
ProcessLog(' End ZigZag Scan ')


ProcessLog(' Start RLC for AC Signal ')
# RLC for AC
Y_rlc = RLC_Zig(Y_zig)
Cb_rlc = RLC_Zig(Cb_zig)
Cr_rlc = RLC_Zig(Cr_zig)
ProcessLog(' End RLC for AC Signal ')


ProcessLog(' Start DPCM for DC Signal ')
# DPCM for DC
Y_dpcm = DPCM(Y_zig)
Cb_dpcm = DPCM(Cb_zig)
Cr_dpcm = DPCM(Cr_zig)
ProcessLog(' End DPCM for DC Signal ')

ProcessLog(' Start Entropy Coding for RLC ')
# Entropy Coding for RLC
# Y_rlc = RLC_Zig(Y_zig)
# Cb_rlc = RLC_Zig(Cb_zig)
# Cr_rlc = RLC_Zig(Cr_zig)
ProcessLog(' End Entropy Coding for RLC ')


ProcessLog(' Start Entropy Coding for DPCM ')
# Entropy Coding for DPCM
# Y_dpcm = DPCM(Y_zig)
# Cb_dpcm = DPCM(Cb_zig)
# Cr_dpcm = DPCM(Cr_zig)
ProcessLog(' End Entropy Coding for DPCM ')


ProcessLog(' Start DeRLC for AC Signal ')
# DeRLC for AC
Y_derlc = DeRLC_zig(Y_rlc)
Cb_derlc = DeRLC_zig(Cb_rlc)
Cr_derlc = DeRLC_zig(Cr_rlc)
ProcessLog(' End DeRLC for AC Signal ')


ProcessLog(' Start DeDPCM for DC Signal ')
# DeDPCM for DC
Y_dedpcm = DeDPCM(Y_dpcm)
Cb_dedpcm = DeDPCM(Cb_dpcm)
Cr_dedpcm = DeDPCM(Cr_dpcm)
ProcessLog(' End DeDPCM for DC Signal ')


ProcessLog(' Start Recover ZigZag from DPCM and RLC ')
# Merge DeDPCM and DeRLC
Y_recover = Merge_RLC_DPCM(Y_derlc, Y_dedpcm)
Cb_recover = Merge_RLC_DPCM(Cb_derlc, Cb_dedpcm)
Cr_recover = Merge_RLC_DPCM(Cr_derlc, Cr_dedpcm)
ProcessLog(' End Recover ZigZag from DPCM and RLC ')


ProcessLog(' Start DeZigZag ')
# DeZigzag scan image
quant_img = DeZigZagImg(quant_img, Y_recover, Cb_recover, Cr_recover)
ProcessLog(' End DeZigZag ')


ProcessLog(' Start DeQuantization ')
# Dequantify image
dequant_img = DeQuantifyImg(quant_img)
ProcessLog(' End DeQuantization ')


ProcessLog(' Start IDCT ')
# IDCT image
idct_img = IDCTImg(dequant_img)
ProcessLog(' End IDCT ')


ProcessLog(' Start Reconstruct Image into RGB')
# reconstruct image
reconstruct_img = YCbCrtoRGBSpace(idct_img)
ProcessLog(' End Reconstruct Image into RGB')


ProcessLog(' Start Calculating Distortion ')
# Distortion Evaluation
Distortion(img_c, reconstruct_img, imgPath)
ProcessLog(' End Calculating Distortion ')


ProcessLog(' Display Result Image')
cv.imshow('my_img', reconstruct_img)
cv.imshow('origin_img', img_c)
cv.waitKey(0)
cv.destroyAllWindows()