

import numpy as np, cv2, pandas as pd, time
import matplotlib.pyplot as plt
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

set_seed(0)
set_context(mode=GRAPH_MODE, device_target="GPU", device_id=0)
print(ms.__version__)


train_data_dir = '/media/hhj/localssd/DL_data/FairFace/fairface-img-margin025-trainval'
test_data_dir = '/media/hhj/localssd/DL_data/FairFace/fairface-img-margin025-trainval'

EPOCH = 20
BATCH_SIZE = 8
IMAGE_SIZE = 224
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


trn_dt = pd.read_csv('/media/hhj/localssd/DL_data/FairFace/fairface_label_train.csv')
print(len(trn_dt))
age_names = sorted(trn_dt['age'].unique())

ages_dict = {}
ages_classes = {}
val = 0
for age in age_names:
    ages_dict[age] = val
    ages_classes[val] = age
    val += 1

print(ages_dict)
print(ages_classes)
gen_classes = {0: "Male", 1: "Female"}


def getFacesData(train_data_dir, trn_dt, num_samples=0):
    imgs, ages, gens = [], [], []

    for ix in range(len(trn_dt)):
        f = trn_dt.iloc[ix].squeeze()
        test_file = train_data_dir + '/' + f.file
        if not os.path.exists(test_file):
            print(test_file + ' ------------------ not exist.')
        else:
            img = cv2.imread(test_file)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 将输入图像的shape从 <H, W, C> 转换为 <C, H, W>
            img = img.transpose(2, 0, 1)/255.
            img = normalize(img)
            img = np.expand_dims(img, axis=0)
            imgs.append(img.astype(np.float32).copy())
            if ages_dict.__contains__(f.age):
                ages.append(ages_dict[f.age])
            else:
                ages.append(9)
            gen = 0
            if f.gender == 'Female':
                gen = 1
            gens.append(gen)
        if num_samples > 0:
            if ix > num_samples:
                break
    return np.vstack(imgs), np.vstack(ages).astype(np.int32), np.vstack(gens).astype(np.int32)


class GenderAgeClass:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def __getitem__(self, ix):
        x, y, z = self.x[ix], self.y[ix], self.z[ix]
        return x, y, z

    def __len__(self):
        return len(self.x)


trn_images, trn_age, trn_gen = getFacesData(train_data_dir, trn_dt, num_samples=8000)
print('images.shape: ', trn_images.shape, ' age.ahspe: ', trn_age.shape, ' ', trn_gen.shape)


tst_dt = pd.read_csv('/media/hhj/localssd/DL_data/FairFace/fairface_label_val.csv')
print(tst_dt.head())
tst_images, tst_age, tst_gen = getFacesData(test_data_dir, tst_dt, num_samples=2000)

trn_dataset = ds.GeneratorDataset(GenderAgeClass(trn_images, trn_age, trn_gen),
                                  column_names=["image", "age", "gen"],
                                  num_parallel_workers=1,
                                  shuffle=True,
                                  sampler=None).batch(BATCH_SIZE, drop_remainder=True)

tst_dataset = ds.GeneratorDataset(GenderAgeClass(tst_images, tst_age, tst_gen),
                                  column_names=["image", "age", "gen"],
                                  num_parallel_workers=1,
                                  shuffle=True,
                                  sampler=None).batch(BATCH_SIZE, drop_remainder=True)

images, ages, gens = tst_dataset.create_tuple_iterator().__next__()
print(images.shape, ' ', ages.shape, ' ', gens.shape)


class ageGenderClassifier(nn.Cell):

    def __init__(self, num_classes):
        super(ageGenderClassifier, self).__init__()
        self.layers = VGG16(imgScale=7, numClasses=1000).layers
        self.num_classes = num_classes

        self.avgpool = nn.SequentialCell(
            nn.Conv2d(512, 512, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Flatten()
        )

        self.intermediate = nn.SequentialCell(
            nn.Dense(2048,512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Dense(512,128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Dense(128,64),
            nn.ReLU(),
        )

        self.age_classifier = nn.SequentialCell(
            nn.Dense(64, self.num_classes)
        )

        self.gender_classifier = nn.SequentialCell(
            nn.Dense(64, 1),
            nn.Sigmoid()
        )

    def construct(self, x):
        x = self.layers(x)
        x = self.avgpool(x)
        x = self.intermediate(x)
        age = self.age_classifier(x)
        gender = self.gender_classifier(x)
        return gender, age


model = ageGenderClassifier( len(ages_classes) )
# ----------------------------------------------------------------
# summary of model
# ----------------------------------------------------------------
X = ops.randn(1, 3, 224, 224)
'''
for blk in model.layers:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)
print('*'*40)
for blk in model.avgpool:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)

print('-'*40)
for blk in model.classifier.intermediate:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)

for blk in model.classifier.age_classifier:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)
'''

gender_criterion = nn.BCELoss()
age_criterion = nn.L1Loss()
loss_fn = gender_criterion, age_criterion
optimizer = nn.Adam(model.trainable_params(), learning_rate=1e-4)


'''
ims, age, gender = (tst_dataset.create_tuple_iterator()).__next__()
print( ims.dtype, ' ', age.dtype, ' ', gender.dtype)
pred_gender, pred_age = model(ims)
print(pred_gender.dtype, ' ', pred_age.dtype, ' ', pred_age.shape)
print(pred_gender.shape, ' ', gender.astype(ms.float32).shape)
print(pred_gender)
print(gender)
gender_loss = gender_criterion(pred_gender, gender.astype(ms.float32))
age_loss = age_criterion(pred_age.argmax(1).astype(ms.float32), age.squeeze().astype(ms.float32))
total_loss = gender_loss + age_loss
print('total_loss: ', total_loss)
print(float(ops.abs(age.squeeze()*1.0/len(ages_classes) - pred_age.argmax(1)*1.0/len(ages_classes)).sum()))
print(age.squeeze(), ' ', pred_age.argmax(1))
pred_age = pred_age.argmax(1)

pred_gender = gender.asnumpy()
pred_age = age.asnumpy()

print(int(pred_age.item()), ' ', pred_gender[0][0])
'''


def train(model, dataset, loss_fn, optimizer):

    gender_criterion, age_criterion = loss_fn

    def forward_fn(ims, age, gender):
        pred_gender, pred_age = model(ims)
        pred_age = pred_age.argmax(1)
        gender_loss = gender_criterion(pred_gender, gender.astype(ms.float32))
        age_loss = age_criterion(pred_age.astype(ms.float32), age.squeeze().astype(ms.float32))
        total_loss = gender_loss + age_loss
        return total_loss

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    def train_step(ims, age, gender):
        loss, grads = grad_fn(ims, age, gender)
        # 获得损失 depend用来处理操作间的依赖关系
        loss = ops.depend(loss, optimizer(grads))
        return loss

    model.set_train()
    train_epoch_losses = 0

    for batch, (ims, age, gender) in enumerate(dataset.create_tuple_iterator()):
        # 批量训练获得损失值
        loss = train_step(ims, age, gender)
        train_epoch_losses += float(loss.item())

    return train_epoch_losses


def validate(dataset, model, loss_fn):
    model.set_train(False)
    gender_criterion, age_criterion = loss_fn
    epoch_test_loss, val_age_mae, val_gender_acc, ctr = 0, 0, 0, 0

    for ix, data in enumerate(dataset):
        ims, age, gender = data
        pred_gender, pred_age = model(ims)
        pred_age = pred_age.argmax(1)
        gender_loss = gender_criterion(pred_gender, gender.astype(ms.float32))
        age_loss = age_criterion(pred_age.astype(ms.float32), age.squeeze().astype(ms.float32))
        total_loss = gender_loss + age_loss
        pred_gender = (pred_gender > 0.5).astype(ms.int32)
        #print('='*50)
        #print(pred_gender.squeeze())
        #print(gender.squeeze())
        gender_acc = float((pred_gender == gender).astype(ms.float32).sum())
        age_mae = float(ops.abs(age - pred_age).sum())
        epoch_test_loss += float(total_loss.item())
        val_gender_acc += gender_acc
        val_age_mae += age_mae
        ctr += ims.shape[0]

    return epoch_test_loss, val_gender_acc, val_age_mae, ctr


print('*'*40, ' Model with all data points ', '*'*40)
val_gender_accuracies = []
val_age_maes = []
train_losses = []
val_losses = []

best_test_loss = 1000
start = time.time()

for t in range(EPOCH):
    print(f"Epoch {t+1} ", '-'*80)
    epoch_train_loss = train(model, trn_dataset, loss_fn, optimizer)

    epoch_test_loss, val_gender_acc, val_age_mae, ctr = validate(tst_dataset, model, loss_fn)

    val_age_mae /= ctr
    val_gender_acc /= ctr
    epoch_train_loss /= len(trn_images)
    epoch_test_loss /= len(tst_images)

    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_test_loss)
    val_gender_accuracies.append(val_gender_acc)
    val_age_maes.append(val_age_mae)

    elapsed = time.time()-start
    best_test_loss = min(best_test_loss, epoch_test_loss)
    print('{}/{} ({:.2f}s - {:.2f}s remaining)'.format(t+1, EPOCH, time.time()-start, (EPOCH-t)*(elapsed/(t+1))))
    info = f"Train Loss: {epoch_train_loss:>.3f}\tTest: {epoch_test_loss:>.3f}\tBest Test Loss: {best_test_loss:>.4f}"
    info += f"\nGender Accuracy: {val_gender_acc*100:>.2f}%\tAge MAE: {val_age_mae:>.2f}\n"
    print(info)


epochs = np.arange(1,len(val_gender_accuracies)+1)
fig,ax = plt.subplots(1,3,figsize=(15,5))
ax = ax.flat
ax[0].plot(epochs, train_losses, 'bo', label='Training')
ax[0].plot(epochs, val_losses, 'r', label='Validation')
ax[0].set_title('Training vs Validation loss')
ax[0].legend()
ax[0].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')
ax[1].plot(epochs, val_gender_accuracies, 'bo')
ax[2].plot(epochs, val_age_maes, 'r')
ax[1].set_xlabel('Epochs')
ax[2].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[2].set_ylabel('MAE')
ax[1].set_title('Validation Gender Accuracy')
ax[2].set_title('Validation Age Mean-Absolute-Error')
plt.show()


def preprocess_image(im):
    im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = ms.Tensor(im).permute(2,0,1)
    im = normalize(im/255.)
    return im[None]


im = cv2.imread('../../data/image/175.jpg')
im = preprocess_image(im)
gender, age = model(im)
age = age.argmax(1)
pred_gender = gender.asnumpy()
pred_age = age.asnumpy()
im = cv2.imread('../../data/image/175.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
label = "{},{}".format(np.where(pred_gender[0][0]<0.5, 'Male', 'Female'), ages_classes[int(pred_age.item())])
cv2.putText(im, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
plt.imshow(im)
plt.show()
print(pred_gender)
print(pred_age)
print('predicted gender: ',
      np.where(pred_gender[0][0]<0.5, 'Male', 'Female'),
      '; Predicted age: ', ages_classes[int(pred_age.item())])


exit(0)