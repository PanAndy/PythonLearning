import cv2
import numpy as np


img = cv2.imread("car.jpg")

height, width = img.shape[0], img.shape[1]
print(width, " ", height)

window = np.outer(np.hanning(height), np.hanning(width))

windows = np.stack((window, window, window), 0)
windows = windows.transpose(1, 2, 0)

add_cosine_window_img = (np.multiply(img, windows)).astype(np.uint8)

cv2.imshow("car window", img)

cv2.imshow("cosine window", window)

cv2.imshow("add cosine window img", add_cosine_window_img)

cv2.waitKey(0)


cv2.destroyAllWindows()


# ROU Net
