import numpy as np
import cv2
import math
from matplotlib import pyplot as plt


def linear_interpolation(img, size):
    shape = img.shape
    result = np.ones((int(shape[0] * size), int(shape[1] * size), shape[2]))
    for i in range(shape[0] * size):
        for j in range(shape[1] * size):


    return result


image = cv2.imread("bee.jpg")
image = image.astype(np.float32) / 255
image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result = linear_interpolation(image1, 4)
result = np.round(result * 255).astype(np.uint8)
plt.imshow(result)
plt.show()