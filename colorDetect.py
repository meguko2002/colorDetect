import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image1.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)
lower_red = np.array([160, 50, 50])
upper_red = np.array([180, 255, 255])
mask += cv2.inRange(hsv, lower_red, upper_red)
# Bitwise-AND mask and original image
res = cv2.bitwise_and(img, img, mask=mask)

image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print("個数 =",len(contours))
print(type(contours))
# 領域の重心を計算
P = np.empty([len(contours),2])
print(type(P))
for i in range(len(contours)):
# for i in contours:
    cnt = contours[i]
    M = cv2.moments(cnt)
    P[i][0] = int(M['m10'] / M['m00'])
    P[i][1] = int(M['m01'] / M['m00'])

    # 検出した領域を表示
    # cv2.drawContours(img, cnt, -1, (0, 255, 0), 3)
print(P)
plt.imshow(res)
plt.colorbar()
plt.show()