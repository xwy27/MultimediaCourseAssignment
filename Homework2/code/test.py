def TEST_DCT_IDCT(img):
  row = img.shape[0]
  col = img.shape[1]
  x = 0
  y = 0
  while (x < row - 7):
    while (y < col - 7):
      print('\n\n')
      y_matrix = createMatrix(8, 8)
      cb_matrix = createMatrix(8, 8)
      cr_matrix = createMatrix(8, 8)
      print('-' * 15, "DCT START", '-' * 15)
      for i in range(0, 8):
        for j in range(0, 8):
          y_matrix[i][j] = img.item(x + i, y + j, CV_YCbCrChannel.Y.value) - 128
          cb_matrix[i][j] = img.item(x + i, y + j, CV_YCbCrChannel.Cb.value) - 128
          cr_matrix[i][j] = img.item(x + i, y + j, CV_YCbCrChannel.Cr.value) - 128
        print(y_matrix[i])
      y_matrix = DCT_1D_COL(y_matrix)
      cb_matrix = DCT_1D_COL(cb_matrix)
      cr_matrix = DCT_1D_COL(cr_matrix)
      
      y_matrix = DCT_1D_ROW(y_matrix)
      cb_matrix = DCT_1D_ROW(cb_matrix)
      cr_matrix = DCT_1D_ROW(cr_matrix)

      print('-' * 15, "DCT DONE", '-' * 15)
      for i in range(0, 8):
        print(y_matrix[i])
        for j in range(0, 8):
          img.itemset((x + i, y + j, CV_YCbCrChannel.Y.value), y_matrix[i][j])
          img.itemset((x + i, y + j, CV_YCbCrChannel.Cb.value), cb_matrix[i][j])
          img.itemset((x + i, y + j, CV_YCbCrChannel.Cr.value), cr_matrix[i][j])
          # print(img.item(x + i, y + j, CV_YCbCrChannel.Y.value))

      y_matrix = IDCT_1D_ROW(y_matrix)
      cb_matrix = IDCT_1D_ROW(cb_matrix)
      cr_matrix = IDCT_1D_ROW(cr_matrix)
      
      y_matrix = IDCT_1D_COL(y_matrix)
      cb_matrix = IDCT_1D_COL(cb_matrix)
      cr_matrix = IDCT_1D_COL(cr_matrix)
      print('-' * 15, "IDCT DONE", '-' * 15)
      for i in range(0, 8):
        print(y_matrix[i])
        for j in range(0, 8):
          img.itemset((x + i, y + j, CV_YCbCrChannel.Y.value), y_matrix[i][j])
          img.itemset((x + i, y + j, CV_YCbCrChannel.Cb.value), cb_matrix[i][j])
          img.itemset((x + i, y + j, CV_YCbCrChannel.Cr.value), cr_matrix[i][j])

      y += 8
    x += 8
  return img

def TEST_QUANTIFY_DEQUANTIFY(img):
  row = img.shape[0]
  col = img.shape[1]
  x = 0
  y = 0
  while (x < row - 7):
    print('\n\n')
    while (y < col - 7):
      y_matrix = createMatrix(8, 8)
      cb_matrix = createMatrix(8, 8)
      cr_matrix = createMatrix(8, 8)
      print(15 * '-' + 'QUANTIFY START' + 15 * '-')
      for i in range(0, 8):
        for j in range(0, 8):
          y_matrix[i][j] = img.item(x + i, y + j, CV_YCbCrChannel.Y.value)
          cb_matrix[i][j] = img.item(x + i, y + j, CV_YCbCrChannel.Cb.value)
          cr_matrix[i][j] = img.item(x + i, y + j, CV_YCbCrChannel.Cr.value)
        print(y_matrix[i])
      y_matrix = Quantify(y_matrix, QUANTIFYMATRIX.LUMINANCE)
      cb_matrix = Quantify(cb_matrix, QUANTIFYMATRIX.CHROMINANCE)
      cr_matrix = Quantify(cr_matrix, QUANTIFYMATRIX.CHROMINANCE)

      print(15 * '-' + 'QUANTIFY DONE' + 15 * '-')
      for i in range(0, 8):
        print(y_matrix[i])
        for j in range(0, 8):
          img.itemset((x + i, y + j, CV_YCbCrChannel.Y.value), y_matrix[i][j])
          img.itemset((x + i, y + j, CV_YCbCrChannel.Cb.value), cb_matrix[i][j])
          img.itemset((x + i, y + j, CV_YCbCrChannel.Cr.value), cr_matrix[i][j])
      y_matrix = DeQuantify(y_matrix, QUANTIFYMATRIX.LUMINANCE)
      cb_matrix = DeQuantify(cb_matrix, QUANTIFYMATRIX.CHROMINANCE)
      cr_matrix = DeQuantify(cr_matrix, QUANTIFYMATRIX.CHROMINANCE)

      print(15 * '-' + 'DEQUANTIFY DONE' + 15 * '-')
      for i in range(0, 8):
        print(y_matrix[i])
        for j in range(0, 8):
          img.itemset((x + i, y + j, CV_YCbCrChannel.Y.value), y_matrix[i][j])
          img.itemset((x + i, y + j, CV_YCbCrChannel.Cb.value), cb_matrix[i][j])
          img.itemset((x + i, y + j, CV_YCbCrChannel.Cr.value), cr_matrix[i][j])
      y += 8
    x += 8
  return img

# def TEST_DCT_IDCT(img):
#   # test = cv.imread('../img/animal.jpg', cv.IMREAD_COLOR)
#   # test = cv.resize(test,(1000,720),interpolation = cv.INTER_CUBIC)
#   # test = cv.cvtColor(test, cv.COLOR_BGR2YCR_CB)
#   # test = test.astype('float')
#   # img_dct=cv.CreateMat(test.rows,test.cols,cv.CV_32FC1)
#   # test = cv.dct(test, img_dct)
#   row = img.shape[0]
#   col = img.shape[1]
#   x = 0
#   y = 0
#   while (x < row - 7):
#     while (y < col - 7):
#       print('\n\n')
#       y_matrix = createMatrix(8, 8)
#       cb_matrix = createMatrix(8, 8)
#       cr_matrix = createMatrix(8, 8)
#       oy_matrix = createMatrix(8, 8)
#       ocb_matrix = createMatrix(8, 8)
#       ocr_matrix = createMatrix(8, 8)
#       print('-' * 15, "DCT START", '-' * 15)
#       for i in range(0, 8):
#         for j in range(0, 8):
#           y_matrix[i][j] = img.item(x + i, y + j, CV_YCbCrChannel.Y.value)-128
#           cb_matrix[i][j] = img.item(x + i, y + j, CV_YCbCrChannel.Cb.value)-128
#           cr_matrix[i][j] = img.item(x + i, y + j, CV_YCbCrChannel.Cr.value)-128
#           # oy_matrix[i][j] = test.item(x + i, y + j, CV_YCbCrChannel.Y.value)
#           # ocb_matrix[i][j] = test.item(x + i, y + j, CV_YCbCrChannel.Cb.value)
#           # ocr_matrix[i][j] = test.item(x + i, y + j, CV_YCbCrChannel.Cr.value)
#         print(y_matrix[i])
#       y_matrix = DCT_1D_COL(y_matrix)
#       cb_matrix = DCT_1D_COL(cb_matrix)
#       cr_matrix = DCT_1D_COL(cr_matrix)
      
#       y_matrix = DCT_1D_ROW(y_matrix)
#       cb_matrix = DCT_1D_ROW(cb_matrix)
#       cr_matrix = DCT_1D_ROW(cr_matrix)

#       print('-' * 15, "DCT DONE", '-' * 15)
#       for i in range(0, 8):
#         # print('+' * 5, "DCT-My", '+' * 5)
#         print(y_matrix[i])
#         # print('+' * 5, "DCT-Lib", '+' * 5)
#         # print(oy_matrix[i])
#         for j in range(0, 8):
#           img.itemset((x + i, y + j, CV_YCbCrChannel.Y.value), y_matrix[i][j])
#           img.itemset((x + i, y + j, CV_YCbCrChannel.Cb.value), cb_matrix[i][j])
#           img.itemset((x + i, y + j, CV_YCbCrChannel.Cr.value), cr_matrix[i][j])
#           # print(img.item(x + i, y + j, CV_YCbCrChannel.Y.value))
#           y_matrix[i][j] = img.item(x + i, y + j, CV_YCbCrChannel.Y.value)
#           cb_matrix[i][j] = img.item(x + i, y + j, CV_YCbCrChannel.Cb.value)
#           cr_matrix[i][j] = img.item(x + i, y + j, CV_YCbCrChannel.Cr.value)

#       y_matrix = IDCT_1D_ROW(y_matrix)
#       cb_matrix = IDCT_1D_ROW(cb_matrix)
#       cr_matrix = IDCT_1D_ROW(cr_matrix)
      
#       y_matrix = IDCT_1D_COL(y_matrix)
#       cb_matrix = IDCT_1D_COL(cb_matrix)
#       cr_matrix = IDCT_1D_COL(cr_matrix)
#       print('-' * 15, "IDCT DONE", '-' * 15)
#       for i in range(0, 8):
#         print(y_matrix[i])
#         for j in range(0, 8):
#           img.itemset((x + i, y + j, CV_YCbCrChannel.Y.value), y_matrix[i][j])
#           img.itemset((x + i, y + j, CV_YCbCrChannel.Cb.value), cb_matrix[i][j])
#           img.itemset((x + i, y + j, CV_YCbCrChannel.Cr.value), cr_matrix[i][j])

#       y += 8
#     x += 8
#   return img
# cv.imshow('dct', TEST_DCT_IDCT(sample_img))

# row = img.shape[0]
# col = img.shape[1]
# x = 0
# y = 0
# while (x < row - 7):
#   while (y < col - 7):
#     y_matrix = createMatrix(8, 8)
#     cb_matrix = createMatrix(8, 8)
#     cr_matrix = createMatrix(8, 8)
#     oy_matrix = createMatrix(8, 8)
#     ocb_matrix = createMatrix(8, 8)
#     ocr_matrix = createMatrix(8, 8)
#     for i in range(0, 8):
#       for j in range(0, 8):
#         y_matrix[i][j] = idct_img.item(x + i, y + j, CV_YCbCrChannel.Y.value)
#         cb_matrix[i][j] = idct_img.item(x + i, y + j, CV_YCbCrChannel.Cb.value)
#         cr_matrix[i][j] = idct_img.item(x + i, y + j, CV_YCbCrChannel.Cr.value)
#         oy_matrix[i][j] = img_copy.item(x + i, y + j, CV_YCbCrChannel.Y.value)
#         ocb_matrix[i][j] = img_copy.item(x + i, y + j, CV_YCbCrChannel.Cb.value)
#         ocr_matrix[i][j] = img_copy.item(x + i, y + j, CV_YCbCrChannel.Cr.value)
#       print(15 * '-' + 'My-Y' + 15 * '-')
#       print(y_matrix[i])
#       print(15 * '-' + 'Origin-Y' + 15 * '-')
#       print(oy_matrix[i])
#       print(15 * '-' + 'My-Cb' + 15 * '-')
#       print(cb_matrix[i])
#       print(15 * '-' + 'Origin-Cb' + 15 * '-')
#       print(ocb_matrix[i])
#       print(15 * '-' + 'My-Cr' + 15 * '-')
#       print(cr_matrix[i])
#       print(15 * '-' + 'Origin-Cr' + 15 * '-')
#       print(ocr_matrix[i])
    
#     y += 8
#   x += 8

def TEST_ZIG_DEZIG(img):
  Y_zig = []
  Cb_zig = []
  Cr_zig = []
  row = img.shape[0]
  col = img.shape[1]
  x = 0
  y = 0
  t = 0
  while (x < row - 7):
    while (y < col - 7):
      y_matrix = createMatrix(8, 8)
      cb_matrix = createMatrix(8, 8)
      cr_matrix = createMatrix(8, 8)
      print(15 * '-' + 'ZIGZAG START' + 15 * '-')
      for i in range(0, 8):
        for j in range(0, 8):
          y_matrix[i][j] = img.item(x + i, y + j, CV_YCbCrChannel.Y.value)
          cb_matrix[i][j] = img.item(x + i, y + j, CV_YCbCrChannel.Cb.value)
          cr_matrix[i][j] = img.item(x + i, y + j, CV_YCbCrChannel.Cr.value)
        print y_matrix[i]
      Y_zig.append(ZigZag(y_matrix))
      Cb_zig.append(ZigZag(cb_matrix))
      Cr_zig.append(ZigZag(cr_matrix))
      print(15 * '-' + 'ZIGZAG DONE' + 15 * '-')
      print Y_zig[t]
      print(15 * '-' + 'DEZIGZAG START' + 15 * '-')
      y_matrix = DeZigZag(Y_zig[t])
      for i in range(0, 8):
        print y_matrix[i]
      t += 1
      y += 8
    x += 8
  return Y_zig, Cb_zig, Cr_zig

def TEST_RLC_DERLC(zig):
  '''
  RLC AC signal in Zig
  '''
  zig_rlc = []
  dezig = []
  for i in range(0, len(zig)):
    print(15 * '-' + 'RLC START' + 15 * '-')
    print zig[i]
    zig_rlc.append(RLC(zig[i]))
    print(15 * '-' + 'RLC DONE' + 15 * '-')
    print zig_rlc[i]
    print(15 * '-' + 'DERLC START' + 15 * '-')
    dezig.append(DeRLC(zig_rlc[i]))
    print dezig[i]
  return zig_rlc