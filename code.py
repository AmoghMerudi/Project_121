import cv2 
import time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*"XVID")
outputFile = cv2.VideoWriter("output.avi", fourcc, 20, (640, 480))
cap = cv2.VideoCapture(0)

time.sleep(2)
bg = 0

for i in range(60):
    ret, bg = cap.read()

bg = np.flip(bg, axis = 1)

while(cap.isOpened()):
    ret, img = cap.read()

    if not ret:
        break
    
    img = np.flip(img, axis = 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_black = np.array([104, 153, 70])
    upper_black = np.array([30, 30, 0])

    mask1 = cv2.inRange(hsv, lower_black, upper_black)
        
    lower_black = np.array([104, 153, 70])
    upper_black = np.array([30, 30, 0])

    mask2 = cv2.inRange(hsv, lower_black, upper_black)

    mask1 = mask1 + mask2

    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    mask2 = cv2.bitwise_not(mask1)

    resolution1 = cv2.bitwise_and(img, img, mask = mask2)
    resolution2 = cv2.bitwise_and(bg, bg, mask = mask1)

    finalOutput = cv2.addWeighted(resolution1, 1, resolution2, 1, 0)
    outputFile.write(finalOutput)

    cv2.imshow("magic", finalOutput)
    cv2. waitKey(1)

cap.release()
out.release()
cv2.destroyAllWindows()
