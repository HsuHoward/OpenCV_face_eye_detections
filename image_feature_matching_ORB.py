import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('bottle.jpg', 1)
img2 = cv2.imread('bottle_mix.jpg', 1)

# ORB method for feature matching. use CV2.SIFT() for SIFT method.
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)
# sort the matches according to the distance
# lambda effectively creates an inline function. input x; output x.distance
matches = sorted(matches, key=lambda x: x.distance)

# draw first 10 matches
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

cv2.imshow('match', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
