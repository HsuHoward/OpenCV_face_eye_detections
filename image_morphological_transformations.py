import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hav = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # color filtering
    # hav hue sat value
    lower_red = np.array([80, 0, 0])
    upper_red = np.array([255, 255, 255])

    mask = cv2.inRange(hav, lower_red, upper_red)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # morphological transformations
    kernel = np.ones((5, 5), np.uint8)
    # erosion- unless all the pixels in the kernel is 1 will leave (1), or will be 0
    # pros- good to erase small noises
    erosion = cv2.erode(mask, kernel, iterations=1)
    # dilation- at least one pixel in the kernel is 1, end up will be 1
    # pros- up-size an object; usually conduct with de-noised image; good for link a segmented image
    dilation = cv2.dilate(mask, kernel, iterations=1)
    # opening- first erosion then dilation
    # pros- good to de-noise a background
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # closing- first dilation then erosion
    # pros- good to de-noise an object
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Morphological gradient- dilation-erosion
    gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    # Top hat/white top-hot- dilation-original
    # pros- make the lightness in the black brighter
    tophat = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernel)
    # Black hat/Black top-hot- original-closing
    # pros- make the darkness in the black darker
    blackhat = cv2.morphologyEx(mask, cv2.MORPH_BLACKHAT, kernel)

    # show image
    cv2.imshow('frame', frame)
    cv2.imshow('res', res)
    cv2.imshow('erosion', erosion)
    cv2.imshow('dilation', dilation)
    cv2.imshow('opening', opening)
    cv2.imshow('closing', closing)
    cv2.imshow('gradient', gradient)
    cv2.imshow('tophat', tophat)
    cv2.imshow('blackhat', blackhat)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
