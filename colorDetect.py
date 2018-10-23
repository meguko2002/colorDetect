import numpy as np
import cv2
def color_pick(color,image):
    if color == 1:        # 青色の検出
        # HSV色空間に変換
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_min = np.array([100,60,150])
        hsv_max = np.array([130,255,255])
        mask = cv2.inRange(hsv, hsv_min, hsv_max)
    elif color == 2:        # 緑色の検出
        # HSV色空間に変換
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_min = np.array([20,60,50])
        hsv_max = np.array([100,255,255])
        mask = cv2.inRange(hsv, hsv_min, hsv_max)
    elif color == 3:        # 赤色の検出
        # HSV色空間に変換
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_min = np.array([0,60,65])
        hsv_max = np.array([10,255,255])
        mask = cv2.inRange(hsv, hsv_min, hsv_max)
        hsv_min = np.array([170,60,65])
        hsv_max = np.array([180,255,255])
        mask += cv2.inRange(hsv, hsv_min, hsv_max)
    return mask

def detect_center_of_gravity(cnt):
    try:
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return cx, cy
    except ZeroDivisionError:
        # たまにゼロ割になってしまうケースが有るので対処
        print("ZeroDivisionError!!")


color = 3  #1=青,2=緑,3=赤
cap = cv2.VideoCapture(0)
kernel = np.ones((5,5),np.uint8)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    mask = color_pick(color, frame)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # print("個数 =", len(contours))

    # 最大の領域を選定
    max_area = 0
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if max_area < area:
            best_cnt = cnt
            max_area = area

    cv2.imshow('mask',mask)
    cv2.imshow('opening',opening)
    imC = frame.copy()
    imC = cv2.drawContours(imC, [best_cnt],0, (0, 255, 0), 3)
    cx, cy = detect_center_of_gravity(best_cnt)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(imC, str(cx)+","+str(cy), (cx, cy), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('origin',imC)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()