
import numpy as np, cv2, pandas as pd, time
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import copy, os
from mindspore import context, set_seed
from mindspore import nn, ops
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV
from mindspore import dtype as mstype
from glob import glob
import sys
sys.path.insert(0, '../utils')
from plotResults import showLossAndAccuracy

set_seed(0)
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=0, save_graphs=False)

classIds = pd.read_csv('/media/hhj/localssd/DL_data/GTSRB/signnames.csv')
classIds.set_index('ClassId', inplace=True)
classIds = classIds.to_dict()['SignName']
cls2int = {f'{k:05d}':ix for ix,(k,v) in enumerate(classIds.items())}
classIds = {f'{k:05d}':v for k,v in classIds.items()}
id2int = {v:ix for ix,(k,v) in enumerate(classIds.items())}
print(id2int)
print(classIds)
print(cls2int)
'''
变量定义
'''
IMAGE_SIZE = 32
BATCH_SIZE = 32
num_epochs = 20

synsets = [
    "00000", "00001", "00002", "00003", "00004", "00005", "00006", "00007", "00008", "00009",
    "00010", "00011", "00012", "00013", "00014", "00015", "00016", "00017", "00018", "00019",
    "00020", "00021", "00022", "00023", "00024", "00025", "00026", "00027", "00028", "00029",
    "00030", "00031", "00032", "00033", "00034", "00035", "00036", "00037", "00038", "00039",
    "00040", "00041", "00042"]

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


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


def getGTSRBData(data_path, cls2int, has_aug=False, num_samples=None):
    de_dataset = ds.ImageFolderDataset(data_path, shuffle=False,
                                       extensions=[".ppm"],
                                       num_samples = num_samples,
                                       num_parallel_workers=1,
                                       class_indexing=cls2int)

    trn_ds, val_ds = de_dataset.split([0.75, 0.25])
    print(len(trn_ds), ' ', len(val_ds))
    augfmt = [
        CV.Decode(to_pil=False),
        CV.Resize([IMAGE_SIZE, IMAGE_SIZE]),
        CV.CenterCrop([IMAGE_SIZE, IMAGE_SIZE]),
        CV.RandomColorAdjust(brightness=(0.8,1.2),
                         contrast=(0.8,1.2),
                         saturation=(0.8,1.2),
                         hue=0.25),
        CV.RandomAffine(5, translate=(0.01,0.1)),
        # 基于给定的缩放和平移因子调整图像的像素大小。输出图像的像素大小为：output = image * rescale + shift。
        # 此处rescale取1.0 / 255.0，shift取0
        CV.Rescale(1.0 / 255.0, 0),
        CV.Normalize(mean=mean, std=std),
        #转换输入图像；形状（H, W, C）为形状（C, H, W）。
        CV.HWC2CHW(),
    ]

    noaugfmt = [
        CV.Decode(to_pil=False),
        CV.Resize([IMAGE_SIZE, IMAGE_SIZE]),
        CV.CenterCrop([IMAGE_SIZE, IMAGE_SIZE]),
        # 基于给定的缩放和平移因子调整图像的像素大小。输出图像的像素大小为：output = image * rescale + shift。
        # 此处rescale取1.0 / 255.0，shift取0
        CV.Rescale(1.0 / 255.0, 0),
        CV.Normalize(mean=mean, std=std),
        #转换输入图像；形状（H, W, C）为形状（C, H, W）。
        CV.HWC2CHW(),
    ]

    #转换为给定MindSpore数据类型的Tensor操作。
    type_cast_op = C.TypeCast(mstype.float32)

    #将操作中的每个操作应用到此数据集。
    if has_aug:
        trn_ds = trn_ds.map(input_columns="image", num_parallel_workers=1, operations=augfmt)
    else:
        trn_ds = trn_ds.map(input_columns="image", num_parallel_workers=1, operations=noaugfmt)
    trn_ds = trn_ds.map(input_columns="image", operations=type_cast_op, num_parallel_workers=1)
    trn_ds = trn_ds.shuffle(buffer_size=len(trn_ds))
    trn_ds = trn_ds.batch(BATCH_SIZE, drop_remainder=True)

    val_ds = val_ds.map(input_columns="image", num_parallel_workers=1, operations=noaugfmt)
    val_ds = val_ds.map(input_columns="image", operations=type_cast_op, num_parallel_workers=1)
    val_ds = val_ds.shuffle(buffer_size=len(val_ds))
    val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=True)

    return trn_ds, val_ds


data_path = '/media/hhj/localssd/DL_data/GTSRB/Train'
trn_ds, val_ds = getGTSRBData(data_path, cls2int, has_aug=False)

batch_num = trn_ds.get_dataset_size()
print('batch_num：', batch_num)
print('训练数据集数量：', trn_ds.get_dataset_size()*BATCH_SIZE)
print('测试数据集数量：', val_ds.get_dataset_size()*BATCH_SIZE)

d_test = copy.deepcopy(val_ds)
data_next=d_test.create_dict_iterator(output_numpy=False).__next__()

images = data_next["image"]
labels = data_next["label"]
print(f"Image shape: {images.shape}, Label: {labels.shape}")
print(images.dtype, ' ', labels.dtype)
print(type(images), ' ', type(labels))


plt.figure(figsize=(16, 8))
img_num = 8
for i in range(img_num):
    plt.subplot(2, img_num // 2, i+1)
    image_trans = denormalize(images[i])
    image_trans = np.transpose(image_trans*255, (1, 2, 0)).astype(np.uint8)
    plt.title(f"{classIds[synsets[int(labels[i].item())]]}")
    plt.imshow(image_trans.asnumpy())
    plt.axis("off")
plt.show()

# ===================================================================================
# no aug and no bn
# ===================================================================================
def convBlock(ni, no):
    return nn.SequentialCell(
        nn.Dropout(p=0.2),
        nn.Conv2d(ni, no, kernel_size=3, pad_mode='pad', padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


class SignClassifier(nn.Cell):
    def __init__(self):
        super().__init__()
        self.model = nn.SequentialCell(
            convBlock(3, 64),
            convBlock(64, 64),
            convBlock(64, 128),
            convBlock(128, 64),
            nn.Flatten(),
            nn.Dense(256, 256),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Dense(256, len(id2int))
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def construct(self, x):
        return self.model(x)

    def compute_metrics(self, preds, targets):
        ce_loss = self.loss_fn(preds, targets)
        acc = (ops.max(preds, 1)[1] == targets).astype(ms.float32).mean()
        return ce_loss, acc


def train_and_val(trn_ds, val_ds, net, opt, num_epochs, tlt):

    criterion = net.compute_metrics

    #前向传播，计算loss
    def forward_fn(inputs, targets):
        logits = net(inputs)
        loss, acc = criterion(logits, targets.flatten())
        return loss, acc


    #计算梯度和loss
    grad_fn = ops.value_and_grad(forward_fn, None, opt.parameters, has_aux=True)


    def train_batch(inputs, targets):
        (loss, acc), grads = grad_fn(inputs, targets)
        loss = ops.depend(loss, opt(grads))
        return loss.item(), acc.item()


    def validate_batch(model, ims, labels, criterion):
        _preds = model(ims)
        loss, acc = criterion(_preds, labels.flatten())
        return loss.item(), acc.item()

    # 创建迭代器
    trn_dl = trn_ds.create_tuple_iterator(num_epochs=num_epochs)
    val_dl = val_ds.create_tuple_iterator(num_epochs=num_epochs)

    # 开始循环训练
    print("Start Training Loop ...")

    trn_losses, trn_acces, val_losses, val_acces = [], [], [], []
    for epoch in range(num_epochs):

        losses, acces = [], []
        # 为每轮训练读入数据
        net.set_train(True)
        for i, (images, labels) in enumerate(trn_dl):
            loss, acc = train_batch(images, labels)
            losses.append(float(loss))
            acces.append(float(acc))

        avg_loss = np.array(losses).mean()
        avg_acc = np.array(acces).mean()
        trn_losses.append(avg_loss)
        trn_acces.append(avg_acc)

        losses, acces = [], []
        net.set_train(False)
        for i, (images, labels) in enumerate(val_dl):
            loss, acc = validate_batch(net, images, labels, criterion)
            losses.append(float(loss))
            acces.append(float(acc))

        avg_loss = np.array(losses).mean()
        avg_acc = np.array(acces).mean()
        val_losses.append(avg_loss)
        val_acces.append(avg_acc)

        print('Epoch: [%3d/%3d], train avg loss: [%5.3f], train avg acc: [%5.3f], '\
            'val avg loss: [%5.3f], val avg acc: [%5.3f]'%(
            epoch+1, num_epochs, trn_losses[epoch], trn_acces[epoch], val_losses[epoch], val_acces[epoch]))

        if epoch == 10: opt = nn.Adam(net.trainable_params(), learning_rate=1e-4)

    showLossAndAccuracy(trn_losses, trn_acces, val_losses, val_acces, num_epochs, tlt)


net = SignClassifier()
opt = nn.Adam(net.trainable_params(), learning_rate=1e-3)

# ----------------------------------------------------------------
# summary of model
# ----------------------------------------------------------------
X = ops.randn(1, 3, 32, 32)
for blk in net.model:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)

train_and_val(trn_ds, val_ds, net, opt, num_epochs, 'no-aug-no-bn')

# ===================================================================================
# no aug and yes bn
# ===================================================================================
def convBlock(ni, no):
    return nn.SequentialCell(
        nn.Dropout(p=0.2),
        nn.Conv2d(ni, no, kernel_size=3, pad_mode='pad', padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(no),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

class SignClassifier(nn.Cell):
    def __init__(self):
        super().__init__()
        self.model = nn.SequentialCell(
            convBlock(3, 64),
            convBlock(64, 64),
            convBlock(64, 128),
            convBlock(128, 64),
            nn.Flatten(),
            nn.Dense(256, 256),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Dense(256, len(id2int))
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def construct(self, x):
        return self.model(x)

    def compute_metrics(self, preds, targets):
        ce_loss = self.loss_fn(preds, targets)
        acc = (ops.max(preds, 1)[1] == targets).astype(ms.float32).mean()
        return ce_loss, acc


trn_ds, val_ds = getGTSRBData(data_path, cls2int, has_aug=False)
net = SignClassifier()
opt = nn.Adam(net.trainable_params(), learning_rate=1e-3)

# ----------------------------------------------------------------
# summary of model
# ----------------------------------------------------------------
X = ops.randn(1, 3, 32, 32)
for blk in net.model:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)

train_and_val(trn_ds, val_ds, net, opt, num_epochs, 'no-aug-yes-bn')

# ===================================================================================
# yes aug and yes bn
# ===================================================================================
trn_ds, val_ds = getGTSRBData(data_path, cls2int, has_aug=True)

net = SignClassifier()
opt = nn.Adam(net.trainable_params(), learning_rate=1e-3)

# ----------------------------------------------------------------
# summary of model
# ----------------------------------------------------------------
X = ops.randn(1, 3, 32, 32)
for blk in net.model:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)

train_and_val(trn_ds, val_ds, net, opt, num_epochs, 'yes-aug-yes-bn')

exit(0)
