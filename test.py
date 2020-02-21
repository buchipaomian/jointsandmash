import cv2
a = cv2.imread("test_mask.png",cv2.IMREAD_UNCHANGED)
b = a[1249:,1273:2641]
cv2.imwrite("cut.png",b)