import numpy as np
import cv2

orig = cv2.imread("siva.png", 1)
img = cv2.resize(orig, (0,0), fx=0.2, fy=0.2)

img1 = img.copy()
img2 = img.copy()

for index, pixel in enumerate(img1):
  print(index)

  if index % 2 == 0:
    img1[index] = [0,0,0]
  else:
    img2[index] = [0,0,0]

cv2.imshow("Subnet 1 input", img1)
cv2.imshow("Subnet 2 input", img2)
cv2.waitKey(0)