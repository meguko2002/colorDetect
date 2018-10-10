import cv2
import numpy as np
import matplotlib.pyplot as plt
# 設定
color = 3# 検出する色を指定（1=青,2=緑,3=赤）

def color_pick(color):
    if color == 1:
        # HSV色空間に変換
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 青色の検出
        hsv_min = np.array([110,150,150])
        hsv_max = np.array([130,255,255])
        mask = cv2.inRange(hsv, hsv_min, hsv_max)
    elif color == 2:
        # HSV色空間に変換
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 緑色の検出
        hsv_min = np.array([30,50,50])
        hsv_max = np.array([80,255,255])
        mask = cv2.inRange(hsv, hsv_min, hsv_max)
    elif color == 3:
        # HSV色空間に変換
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 赤色の検出
        hsv_min = np.array([0,65,65])
        hsv_max = np.array([10,255,255])
        mask = cv2.inRange(hsv, hsv_min, hsv_max)
        hsv_min = np.array([170,65,65])
        hsv_max = np.array([180,255,255])
        mask += cv2.inRange(hsv, hsv_min, hsv_max)
    return mask

img = cv2.imread('image1.jpg')

# Bitwise-AND mask and original image
mask = color_pick(color)
res = cv2.bitwise_and(img, img, mask=mask)

image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print("個数 =",len(contours))
# 領域の重心を計算
P = np.empty([len(contours),2])

for i in range(len(contours)):
    cnt = contours[i]
    try:
        M = cv2.moments(cnt)
        P[i][0] = int(M['m10'] / M['m00'])
        P[i][1] = int(M['m01'] / M['m00'])
        plt.subplot(1, 2, 2), plt.text(P[i][0], P[i][1], P[i], color='g')
    except ZeroDivisionError:
        # たまにゼロ割になってしまうケースが有るので対処
        print("ZeroDivisionError!!")

print(P)

plt.subplot(1,2,1),plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.subplot(1,2,2),plt.imshow(cv2.cvtColor(res,cv2.COLOR_BGR2RGB))
plt.show()
