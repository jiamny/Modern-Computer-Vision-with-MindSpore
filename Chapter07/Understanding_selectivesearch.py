

import selectivesearch
import cv2, numpy as np
from skimage.segmentation import felzenszwalb
import matplotlib.pyplot as plt

img = cv2.imread('../../data/image/Hemanvi.jpeg', 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
segments_fz = felzenszwalb(img, scale=200)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.grid(False)
plt.imshow(img)
plt.title('Original Image')


plt.subplot(122)
plt.grid(False)
plt.imshow(segments_fz)
plt.title('Felzenszwalb segmentation image')
plt.show()



def extract_candidates(img):
    img_lbl, regions = selectivesearch.selective_search(img, scale=200, min_size=100)
    img_area = np.prod(img.shape[:2])
    candidates = []
    for r in regions:
        if r['rect'] in candidates: continue
        if r['size'] < (0.05*img_area): continue
        if r['size'] > (1*img_area): continue
        x, y, w, h = r['rect']
        candidates.append(list(r['rect']))
    return candidates


img = cv2.imread('../../data/image/Hemanvi.jpeg', 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
candidates = extract_candidates(img)
print(candidates)

imOut = img.copy()

# itereate over all the region proposals
for i, rect in enumerate(candidates):
    # draw rectangle for region proposal till numShowRects
    x, y, w, h = rect
    cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)

plt.figure(figsize=(8, 5))
plt.imshow(imOut)
plt.title('Segmentation region proposals')
plt.show()

exit(0)