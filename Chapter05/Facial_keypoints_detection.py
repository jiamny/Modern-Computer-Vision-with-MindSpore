import os
import mindspore as ms
from mindspore import Tensor, nn, set_context, GRAPH_MODE, train, set_seed
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import ops
import matplotlib.pyplot as plt

from vgg16 import VGG16
import cv2
import numpy as np, pandas as pd
from copy import deepcopy

set_seed(0)
set_context(mode=GRAPH_MODE, device_target="GPU", device_id=0)
print(ms.__version__)

train_data_dir = '/media/hhj/localssd/DL_data/P1_Facial_Keypoints/training'
test_data_dir = '/media/hhj/localssd/DL_data/P1_Facial_Keypoints/test'

EPOCH = 10
BATCH_SIZE = 8
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def normalize(img):
    img[0,:,:] = (img[0,:,:] - mean[0])/std[0]
    img[1,:,:] = (img[1,:,:] - mean[1])/std[1]
    img[2,:,:] = (img[2,:,:] - mean[2])/std[2]
    return img


def denormalize(img):
    img[0,:,:] = img[0,:,:]*std[0] + mean[0]
    img[1,:,:] = img[1,:,:]*std[1] + mean[1]
    img[2,:,:] = img[2,:,:]*std[2] + mean[2]
    return img


def getFacesData(train_data_dir, trn_dt):
    imgs = []
    kps = []

    for ix in range(len(trn_dt)):
        test_file = train_data_dir + '/' + trn_dt.iloc[ix,0]
        if not os.path.exists(test_file):
            print(test_file + ' ------------------ not exist.')
        else:
            img = cv2.imread(test_file)
            img = cv2.resize(img, (224,224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            kp = deepcopy(trn_dt.iloc[ix,1:].tolist())
            kp_x = (np.array(kp[0::2])/img.shape[1]).tolist()
            kp_y = (np.array(kp[1::2])/img.shape[0]).tolist()
            kp2 = np.array(kp_x + kp_y).astype(np.float32)
            # 将输入图像的shape从 <H, W, C> 转换为 <C, H, W>
            img = img.transpose(2, 0, 1)/255.
            img = normalize(img)
            img = np.expand_dims(img, axis=0)
            imgs.append(img.copy())
            kps.append(kp2.reshape(-1, kp2.shape[0]))

    images = np.vstack(imgs)
    kpss = np.vstack(kps)
    return images, kpss


class FacesData:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __getitem__(self, ix):
        x, y = self.x[ix], self.y[ix]
        return x, y

    def __len__(self):
        return len(self.x)


trn_dt = pd.read_csv('/media/hhj/localssd/DL_data/P1_Facial_Keypoints/training_frames_keypoints.csv')
trn_images, trn_kpss = getFacesData(train_data_dir, trn_dt)
print('images.shape: ', trn_images.shape, ' kpss.ahspe: ', trn_kpss.shape)

tst_dt = pd.read_csv('/media/hhj/localssd/DL_data/P1_Facial_Keypoints/test_frames_keypoints.csv')
print(tst_dt.head())
tst_images, tst_kpss = getFacesData(test_data_dir, tst_dt)

trn_dataset = ds.GeneratorDataset(FacesData(trn_images, trn_kpss),
                                 column_names=["image", "label"],
                                 num_parallel_workers=1,
                                 shuffle=True,
                                 sampler=None).batch(BATCH_SIZE, drop_remainder=True)

tst_dataset = ds.GeneratorDataset(FacesData(tst_images, tst_kpss),
                                  column_names=["image", "label"],
                                  num_parallel_workers=1,
                                  shuffle=False,
                                  sampler=None).batch(BATCH_SIZE, drop_remainder=True)

images, label = tst_dataset.create_tuple_iterator().__next__()
print(images.shape, ' ', label.shape)
print(type(label))

'''
im = denormalize(images[0])
im = ((im*255).permute((1,2,0)).asnumpy()).astype(np.uint8)
print(type(im))
plt.imshow(im)
plt.show()


# cv2.imread() image in RGB format, needs to convert to RGB format to show in matplotlib
test_file = train_data_dir + '/' + trn_dt.iloc[0,0]
img = cv2.imread(test_file)
img = cv2.resize(img, (224,224))
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
plt.imshow(img2)
plt.show()
'''


model = VGG16(imgScale=224 // 32, numClasses=136)

# ----------------------------------------------------------------
# summary of model
# ----------------------------------------------------------------
X = ops.randn(8, 3, 224, 224)
for blk in model.cells():
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)

loss_fn = nn.L1Loss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=1e-4)

def validate(model, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    test_loss = 0
    for data, label in dataset.create_tuple_iterator():
        pred = model(data.astype(ms.float32))
        test_loss += loss_fn(pred, label.astype(ms.float32)).asnumpy()
    test_loss /= num_batches
    return test_loss


def train(model, dataset, loss_fn, optimizer):

    def forward_fn(data, label):
        pred = model(data.astype(ms.float32))
        loss = loss_fn(pred, label.astype(ms.float32))
        return loss, pred

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        # 获得损失 depend用来处理操作间的依赖关系
        loss = ops.depend(loss, optimizer(grads))
        return loss

    model.set_train()
    train_epoch_losses = []

    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        # 批量训练获得损失值
        loss = train_step(data, label)
        train_epoch_losses.append(float(loss.item()))

    train_epoch_loss = np.array(train_epoch_losses).mean()
    return train_epoch_loss


print('*'*40, ' Model with all data points ', '*'*40)
train_losses, val_losses = [], []

for t in range(EPOCH):
    print(f"Epoch {t+1} ", '-'*80)
    loss = train(model, trn_dataset, loss_fn, optimizer)
    train_losses.append(loss)
    test_loss = validate(model, tst_dataset, loss_fn)
    val_losses.append(test_loss)
    print(f"Train avg loss: {loss:>8f} Test avg loss: {test_loss:>8f}\n")

epochs = np.arange(EPOCH)+1

plt.plot(epochs, train_losses, 'bo', label='Training loss')
plt.plot(epochs, val_losses, 'r', label='Test loss')
plt.title('Training and Test loss over increasing epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid('off')
plt.show()

ix = 0
plt.figure(figsize=(10,5))
plt.subplot(221)
plt.title('Original image')
im = denormalize(tst_images[ix])
im = ((im*255).transpose(1,2,0)).astype(np.uint8)
plt.imshow(im)
plt.grid(False)

plt.subplot(222)
plt.title('Image with facial keypoints')
x = (images[ix]).astype(ms.float32)
plt.imshow(im)
kp = model(x[None]).flatten().asnumpy()
plt.scatter(kp[:68]*224, kp[68:]*224, c='r')
plt.grid(False)
plt.show()

exit(0)