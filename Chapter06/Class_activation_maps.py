# Malaria cell images: https://data.mendeley.com/datasets/y7z2vg7fmy/1

from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import copy
import numpy as np
from mindspore import context, set_seed, get_grad
from mindspore import nn, ops
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV
from mindspore import dtype as mstype
import cv2

set_seed(0)
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=0, save_graphs=False)

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
id2int = {'Parasitized':0, 'Uninfected':1}
classes = ['Parasitized', 'Uninfected']
IMAGE_SIZE = 128
BATCH_SIZE = 8
cfg = edict({
    'num_class': 2,       # 分类类别
    'lr': 1e-3,           # 学习率
    'num_epochs': 2       # 训练次数
})

def denormalize(img):
    img[0,:,:] = img[0,:,:]*std[0] + mean[0]
    img[1,:,:] = img[1,:,:]*std[1] + mean[1]
    img[2,:,:] = img[2,:,:]*std[2] + mean[2]
    return img


def getMalariaImages(data_path, cls2int, num_samples=None):
    de_dataset = ds.ImageFolderDataset(data_path, shuffle=False,
                                       extensions=[".png", ".jpg"],
                                       num_samples = num_samples,
                                       num_parallel_workers=1,
                                       class_indexing=cls2int)

    trn_ds, val_ds = de_dataset.split([0.75, 0.25])
    print(len(trn_ds), ' ', len(val_ds))
    trnfmt = [
        CV.Decode(to_pil=False),
        CV.Resize([IMAGE_SIZE, IMAGE_SIZE]),
        CV.CenterCrop([IMAGE_SIZE, IMAGE_SIZE]),
        CV.RandomColorAdjust(brightness=(0.95,1.05),
                             contrast=(0.95,1.05),
                             saturation=(0.95,1.05),
                             hue=0.05),
        CV.RandomAffine(5, translate=(0.01,0.1)),
        # 基于给定的缩放和平移因子调整图像的像素大小。输出图像的像素大小为：output = image * rescale + shift。
        # 此处rescale取1.0 / 255.0，shift取0
        CV.Rescale(1.0 / 255.0, 0),
        CV.Normalize(mean=mean, std=std),
        #转换输入图像；形状（H, W, C）为形状（C, H, W）。
        CV.HWC2CHW(),
    ]

    valfmt = [
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
    trn_ds = trn_ds.map(input_columns="image", num_parallel_workers=1, operations=trnfmt)
    trn_ds = trn_ds.map(input_columns="image", operations=type_cast_op, num_parallel_workers=1)
    trn_ds = trn_ds.shuffle(buffer_size=len(trn_ds))
    trn_ds = trn_ds.batch(BATCH_SIZE, drop_remainder=True)

    val_ds = val_ds.map(input_columns="image", num_parallel_workers=1, operations=valfmt)
    val_ds = val_ds.map(input_columns="image", operations=type_cast_op, num_parallel_workers=1)
    val_ds = val_ds.shuffle(buffer_size=len(val_ds))
    val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=True)

    return trn_ds, val_ds


set_seed(1234)
data_path = '/media/hhj/localssd/DL_data/cell_images'

(de_train, de_test) = getMalariaImages(data_path, id2int)
batch_num = de_train.get_dataset_size()
print('batch_num：', batch_num)
print('训练数据集数量：',de_train.get_dataset_size()*BATCH_SIZE) #get_dataset_size()获取批处理的大小。
print('测试数据集数量：',de_test.get_dataset_size()*BATCH_SIZE)

d_test = copy.deepcopy(de_test)
data_next=d_test.create_dict_iterator(output_numpy=True).__next__()

images = data_next["image"]
labels = data_next["label"]
print(f"Image shape: {images.shape}, Label: {labels}")
print(images.dtype, ' ', labels.dtype)

plt.figure()
num_images = 8
for i in range(num_images):
    plt.subplot(2, num_images // 2, i+1)
    image_trans = denormalize(images[i])
    image_trans = np.transpose(image_trans*255, (1, 2, 0)).astype(np.uint8)
    plt.title(f"{classes[labels[i]]}")
    plt.imshow(image_trans)
    plt.axis("off")
plt.show()


def convBlock(ni, no):
    return nn.SequentialCell(
        nn.Dropout(p=0.2),
        nn.Conv2d(ni, no, kernel_size=3, pad_mode='pad', padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(no),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


class MalariaClassifier(nn.Cell):
    def __init__(self):
        super().__init__()
        self.model = nn.SequentialCell(
            convBlock(3, 64),
            convBlock(64, 64),
            convBlock(64, 128),
            convBlock(128, 256),
            convBlock(256, 512),
            convBlock(512, 64),
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
        loss = self.loss_fn(preds, targets)
        acc = (ops.max(preds, 1)[1] == targets).astype(ms.float32).mean()
        return loss, acc


model = MalariaClassifier()
opt = nn.Adam(model.trainable_params(), learning_rate=cfg.lr)
criterion = model.compute_metrics
# ----------------------------------------------------------------
# summary of model
# ----------------------------------------------------------------
X = ops.randn(1, 3, 128, 128)

for blk in model.model:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)

'''
X = ops.randn(8, 3, 128, 128)
print('---> ', model(X))
net = model.model[-6][1]
for ix, par in enumerate(net.trainable_params()):
    print(par.mean((1,2,3)).shape)
ppl = net.trainable_params()[0].mean((1,2,3))
print(net.trainable_params()[0].mean((1,2,3)).reshape(1,-1).shape)
print(ppl[2])
#for _, cell in net.cells_and_names():
#    print(cell)
#    print( get_grad(cell.weight) )
#print(model.model[-6][1].get_parameters().grad()) #.weight.grad.data.mean((1,2,3))
'''


#前向传播，计算loss
def forward_fn(inputs, targets):
    logits = model(inputs)
    loss, acc = model.compute_metrics(logits, targets)
    return loss, acc


#计算梯度和loss
grad_fn = ops.value_and_grad(forward_fn, None, opt.parameters, has_aux=True)


def train_batch(inputs, targets):
    (loss, acc), grads = grad_fn(inputs, targets)
    loss = ops.depend(loss, opt(grads))
    return loss.item(), acc.item(), grads


def validate_batch(model, ims, labels, criterion):
    _preds = model(ims)
    loss, acc = criterion(_preds, labels)
    return loss.item(), acc.item()


# 创建迭代器
data_loader_train = de_train.create_tuple_iterator(num_epochs=cfg.num_epochs)
data_loader_val = de_test.create_tuple_iterator(num_epochs=cfg.num_epochs)

# 开始循环训练
print("Start Training Loop ...")

trn_lasses, val_accs = [], []

for epoch in range(cfg.num_epochs):

    losses = []
    # 为每轮训练读入数据
    model.set_train(True)
    for i, (images, labels) in enumerate(data_loader_train):
        loss, acc, _ = train_batch(images, labels)
        losses.append(float(loss))

    model.set_train(False)
    acces = []
    for i, (images, labels) in enumerate(data_loader_val):
        loss, acc = validate_batch(model, images, labels, criterion)
        acces.append(float(acc))

    avg_loss = np.array(losses).mean()
    avg_acc = np.array(acces).mean()

    print('Epoch: [%3d/%3d], Train avg loss: [%5.3f], Val avg acc: [%5.3f]'%(
        epoch+1, cfg.num_epochs, avg_loss, avg_acc))



im2fmap = nn.SequentialCell(*(list(model.model.cells())[:5] + list(list(model.model.cells())[5])[:2]))
print('sumary im2fmap: ')
X = ops.randn(1, 3, 128, 128)

for blk in im2fmap:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)


def im2gradCAM(x):
    model.set_train(False)
    logits = model(x)
    activations = im2fmap(x)
    pred = logits.max(-1, return_indices=True)[-1]

    # get the model's prediction weights
    net = model.model[-6][1]
    pooled_grads = net.trainable_params()[0].mean((1,2,3))
    # multiply each activation map with corresponding gradient average
    for i in range(activations.shape[1]):
        activations[:,i,:,:] *= pooled_grads[i]
    # take the mean of all weighted activation maps
    # (that has been weighted by avg. grad at each fmap)
    heatmap = ops.mean(activations, axis=1)[0].asnumpy()
    return heatmap, 'Uninfected' if pred.item() else 'Parasitized'


SZ = 128
def upsampleHeatmap(map, img):
    m,M = map.min(), map.max()
    map = 255 * ((map-m) / (M-m))
    map = np.uint8(map)
    map = cv2.resize(map, (SZ,SZ))
    map = cv2.applyColorMap(255-map, cv2.COLORMAP_JET)
    map = np.uint8(map)
    map = np.uint8(map*0.7 + img*0.3)
    return map

N = 8

data = d_test.create_dict_iterator(output_numpy=False).__next__()
x = data["image"]
y = data["label"]

for i in range(N):
    image = denormalize(x[i].asnumpy())
    image = np.transpose(image*255, (1, 2, 0)).astype(np.uint8)
    heatmap, pred = im2gradCAM(x[i:i+1])

    if(pred=='Uninfected'):
        continue
    heatmap = upsampleHeatmap(heatmap, image.copy())

    plt.figure(figsize=(8,5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap)
    plt.suptitle(pred)
    plt.axis("off")
    plt.show()


exit(0)
