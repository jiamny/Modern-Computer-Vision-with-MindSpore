import time

import sys
sys.path.insert(0, '../utils')
from fashion_mnist import _FashionMNIST

from imgaug import augmenters as iaa

data_folder = '/media/hhj/localssd/DL_data/fashion_MNIST'
tmnist = _FashionMNIST(data_folder, is_train=True)
tr_images, tr_targets = tmnist.get_data_and_label()

print(tr_images.dtype, ' ', tr_targets.dtype)

val_mnist = _FashionMNIST(data_folder, is_train=False)
val_images, val_targets = val_mnist.get_data_and_label()

start = time.time()
aug = iaa.Sequential([
    iaa.Affine(translate_px={'x':(-10,10)}, mode='constant'),
])
# record end time
end = time.time()

# print the difference between start and end time in milli. secs
print("The time of execution iaa.Sequential :", (end-start) * 10**3, "ms")

start = time.time()
x = aug.augment_images(tr_images[:32])
# record end time
end = time.time()

# print the difference between start and end time in milli. secs
print("The time of execution aug.augment_images :", (end-start) * 10**3, "ms")