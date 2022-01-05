import cv2

img = cv2.imread('sky.jpg', cv2.IMREAD_COLOR)
img = cv2.resize(img, (640, 380))

img[55, 55] = [255, 255, 255]
px = img[55, 55]

img[100:150, 100:150] = [255, 255, 255]

sky_face = img[37:111, 107:194]
img[0:74, 0:87] = sky_face

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
