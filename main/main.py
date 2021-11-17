import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from skimage.filters import threshold_otsu
import random as rng

im = cv.imread('main/images/Image_a5_rice.tif')
grayImg = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

plt.figure()
plt.title('Original Image')
plt.imshow(grayImg, cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.show()

canny_output = cv.Canny(grayImg,100,200)

plt.figure()
plt.title('Canny Edge Detection')
plt.imshow(canny_output, cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.show()

contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours_poly = [None]*len(contours)
boundRect = [None]*len(contours)
for i, c in enumerate(contours):
    contours_poly[i] = cv.approxPolyDP(c, 3, True)
    boundRect[i] = cv.boundingRect(contours_poly[i])

drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

fig = plt.figure(figsize=(14, 10))
rows = 14
columns = 10

plt.suptitle('Extracted Rice Objects', size=15)

for i in range(len(contours)):
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    #color = (255, 255, 255)
    cv.drawContours(drawing, contours_poly, i, color)
    top_left = (int(boundRect[i][0]), int(boundRect[i][1]))
    bottom_right = (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3]))
    
    cv.rectangle(drawing, top_left, bottom_right, color, 2)
    
    tmp_im = im[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]
    fig.add_subplot(rows, columns, i + 1)
    plt.imshow(tmp_im)
    plt.axis('off')

plt.figure()
plt.title('Detected Rice Objects')
plt.imshow(drawing, cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.show()