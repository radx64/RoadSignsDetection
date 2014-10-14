import numpy as np
import cv2

image = cv2.imread('image.png')
gray = cv2.imread('image.png',0)

ret,thresh = cv2.threshold(gray,127,255,1)

contours,h = cv2.findContours(thresh,1,2)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    print "Found " + str(len(approx)) + " corners"
    if len(approx)==3:
        print "triangle"
        cv2.drawContours(image,[cnt],0,(0,255,0),-1)
    elif len(approx)==4:
        print "square"
        cv2.drawContours(image,[cnt],0,(255,0,0),-1)
    elif len(approx) > 4:
        print "circle"
        cv2.drawContours(image,[cnt],0,(0,0,255),-1)

cv2.imshow('Detection result',image)
cv2.waitKey(0)
cv2.destroyAllWindows()