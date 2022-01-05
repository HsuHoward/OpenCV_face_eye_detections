import numpy as np
import cv2

img = cv2.imread('sky.jpg', cv2.IMREAD_COLOR)
img = cv2.resize(img, (640, 380))
# draw something
cv2.line(img, (0, 0), (150, 150), (0, 0, 0), 15)
cv2.rectangle(img, (15, 25), (200, 150), (0, 255, 0), 5)
cv2.circle(img, (100, 63), 55, (0, 0, 255), 10)

pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
# pts = pts.reshape((-1,-1,2))
cv2.polylines(img, [pts], True, (0, 255, 255), 5)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'OpenCV Tuts!', (0, 130), font, 2, (255, 0, 0), 5, cv2.LINE_AA)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
