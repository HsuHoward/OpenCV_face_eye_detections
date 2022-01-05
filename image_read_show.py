import cv2
import numpy as np
import matplotlib.pyplot as plt
#
img = cv2.imread('sky.jpg')
# draw an image
img = cv2.resize(img, (600,600), fx=1, fy=1, interpolation=cv2.IMREAD_GRAYSCALE)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# draw an image with a line
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.plot([50, 100], [80, 100], 'c', linewidth=5)
plt.show()
# save an image
cv2.imwrite('sky.png', img)
