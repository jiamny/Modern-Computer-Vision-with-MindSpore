
import cv2, matplotlib.pyplot as plt

img = cv2.imread('../data/image/Hemanvi.jpeg')
img = img[50:250,40:240]
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray, cmap='gray')
img_gray_small = cv2.resize(img_gray,(25,25))
plt.imshow(img_gray_small, cmap='gray')
plt.show()
print(img_gray_small)

