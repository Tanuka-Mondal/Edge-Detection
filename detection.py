import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("comp.jpg", cv2.IMREAD_GRAYSCALE)
laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
laplacian = np.uint8(np.absolute(laplacian))
sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)
canny = cv2.Canny(image,100,200)

sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

sobelCombined = cv2.bitwise_or(sobelX, sobelY)

names = ['image', 'Laplacian', 'sobelX', 'sobelY', 'sobelCombined', 'Canny']
imgs = [image, laplacian, sobelX, sobelY, sobelCombined, canny]
for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(imgs[i], 'gray')
    plt.title(names[i])
    plt.xticks([]),plt.yticks([])

plt.show()
