
import face_alignment
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io

def read(fname, mode=0):
    img = cv2.imread(str(fname), mode)
    if mode == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
    return img


fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda')
input = io.imread('../../data/image/obama.jpg')
preds = fa.get_landmarks(input)[0]
print(preds.shape)

fig,ax = plt.subplots(figsize=(5,5))
plt.imshow(read('../../data/image/obama.jpg',1)) #, ax=ax)
ax.scatter(preds[:,0], preds[:,1], marker='+', c='r')
plt.show()

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False, device='cpu')
input = read('../../data/image/obama.jpg', 1)
preds = fa.get_landmarks(input)[0]
print(preds.shape)

df = pd.DataFrame(preds)
df.columns = ['x','y','z']

import plotly.express as px
fig = px.scatter_3d(df, x = 'x', y = 'y', z = 'z')
fig.show()

exit(0)
