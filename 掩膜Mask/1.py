import cv2
import numpy as np

sss = np.zeros([480, 640], dtype=np.uint8)
sss[300:350, 310:400] = 255

img0 = cv2.imread("shiyanyi.jpg", 0)
image1 = cv2.add(img0, np.zeros(np.shape(img0), dtype=np.uint8), mask=sss)


cv2.imshow("sss")











