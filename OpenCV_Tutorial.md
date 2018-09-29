# OpenCV

<!-- TOC -->

- [OpenCV](#opencv)
  - [Install](#install)
  - [Image](#image)
    - [Read an image](#read-an-image)
    - [Display an image](#display-an-image)
    - [Write an image](#write-an-image)
    - [Sum up](#sum-up)
    - [Matplotlib](#matplotlib)
    - [Pixel Operation](#pixel-operation)
      - [Access & Modify Pixel](#access--modify-pixel)
      - [Image Properties](#image-properties)

<!-- /TOC -->

## Install

OpenCV supports windows GUI installation or pip.

[**Install Tutorial**](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_setup/py_table_of_contents_setup/py_table_of_contents_setup.html#py-table-of-content-setup)

Following tutorial based on `python` language.

## Image

### Read an image

```python
import numpy as np
import cv2

# Read image
#   image path: 
#     Ignored if image is in the working directory, otherwise, the full path
#   flag:
#     cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected. Default
#     cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode
#     cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel
img = cv2.imread('image path', flag)
```

### Display an image

```python
# Display image
#   string:
#     window name
#   img:
#     image file
cv2.imshow('string', img)

# Keyboard binding function, waits for specified milliseconds for any key board event. If pressed key in that time, program continues.
#   num:
#     time in milliseconds
cv2.waitKey(num)

# Destroy all windows created.
# cv2.destroyWindow(window_name) could destroy specific window named window_name
cv2.destroyAllWindows()
```

### Write an image

```python
# Save an image to the working directory
#   name:
#     image file name
#   img:
#     image file
cv2.imwrite('name', img)
```

### Sum up

```python
import numpy as np
import cv2

img = cv2.imread('messi5.jpg',0)
cv2.imshow('image',img)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()
```

**If you are using a 64-bit machine, you will have to modify k = cv2.waitKey(0) line as follows : k = cv2.waitKey(0) & 0xFF**

### Matplotlib

```python
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('messi5.jpg',0)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
```

**Color image loaded by OpenCV is in BGR mode. But Matplotlib displays in RGB mode. So color images will not be displayed correctly in Matplotlib if image is read with OpenCV. Please see the exercises for more details. [Solution](https://stackoverflow.com/questions/15072736/extracting-a-region-from-an-image-using-slicing-in-python-opencv/15074748#15074748)**

### Pixel Operation

#### Access & Modify Pixel

Retrieve the pixel by row and column of two dimision matrix and returns the BGR values array or RGB values array which depends the mode of image.

```python
img = cv2.imread('img')
px = img[100, 100]  # access
img[100, 100] = [255, 255, 255] # modify
```

*Above method is too slow which brings numpy into practise. `array.item()` reterieves the item and `array.itemset()` sets the item. Faster way below:*

```python
img.item(10, 10, 2) # access red value at [10, 10]

img.itemset((10, 10, 2), 100) # modify red value at [10, 10]
```

#### Image Properties

- Shape

`img.shape` returns a tuple consists of rows, columns and channels

*If image is grayscale, tuple returned contains only number of rows and columns. So it is a good method to check if loaded image is grayscale or color image.*

- Pixels

`img.size` returns total pixels of image file

- Datatype

`img.dtype` returns the data type of image file