import cv2
import numpy as np

img = cv2.imread('boxes.jpeg')
img2 = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# use Shi-Tomasi for corner detection
# set the parameters
maxCorners = 150
qualityLevel = 0.01
minDistance = 10
blockSize = 3
useHarrisDetector = False
k = 0.04

corners = cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance, None, None, blockSize, useHarrisDetector, k)
corners2 = np.copy(corners)
corners = np.int0(corners)

for corner in corners:
    # flatten to one array
    x, y = corner.ravel()
    # thickness=-1 is filled circle
    cv2.circle(img, (x, y), 3, (0, 0, 255), 1)


# cornerSubPix (sub-pixel level) for more precise
# Set the needed parameters to find the refined corners
winSize = (5, 5)
zeroZone = (-1, -1)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)

# Calculate the refined corner locations
corners2 = cv2.cornerSubPix(gray, corners2, winSize, zeroZone, criteria)


for i in range(corners.shape[0]):
    cv2.circle(img2, (int(corners[i,0,0]), int(corners[i,0,1])), 3, (0,0,255), 1)


cv2.imshow('Corner', img)
cv2.imshow('SubCorner', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.cornerSubPix()
