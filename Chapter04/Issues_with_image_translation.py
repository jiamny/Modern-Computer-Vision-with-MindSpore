import mindspore
from mindspore import nn
from mindspore import ops
import mindspore as ms
import mindspore.dataset as ds
from mindspore.dataset import vision
from mindspore.dataset import transforms
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '../utils')
from fashion_mnist import _FashionMNIST
from plotResults import displayImages

from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=0)


data_folder = '/media/hhj/localssd/DL_data/fashion_MNIST'
tmnist = _FashionMNIST(data_folder, is_train=True)
tr_images, tr_targets = tmnist.get_data_and_label()

fmnist_classes = tmnist.classes()
print('fmnist_classes: ', fmnist_classes)

print(tr_images.dtype, ' ', tr_targets.dtype)

val_mnist = _FashionMNIST(data_folder, is_train=False)
val_images, val_targets = val_mnist.get_data_and_label()

def accuracy(model, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    return correct, test_loss


LEARNING_RATE = 1e-3
EPOCH = 5
BATCH_SIZE = 32

def datapipe(dataset, batch_size):
    image_transforms = [
        # 基于给定的缩放和平移因子调整图像的像素大小。输出图像的像素大小为：output = image * rescale + shift。
        # 此处rescale取1.0 / 255.0，shift取0
        vision.Rescale(1.0 / 255.0, 0)
    ]

    label_transform = transforms.TypeCast(ms.int32)

    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')

    # 将数据集中连续 batch_size 条数据组合为一个批数据
    dataset = dataset.batch(batch_size)
    return dataset

class FMNISTDataset:
    def __init__(self, x, y):
        self.y = y
        x = x.reshape(-1,28*28)
        self.x = x

    def __getitem__(self, ix):
        x, y = self.x[ix], self.y[ix]
        return x, y

    def __len__(self):
        return len(self.x)


print(tr_images.shape)

trn_dataset = ds.GeneratorDataset(FMNISTDataset(tr_images, tr_targets),
                                  column_names=["image", "label"],
                                  num_parallel_workers=1,
                                  shuffle=True,
                                  sampler=None)

val_dataset = ds.GeneratorDataset(FMNISTDataset(val_images, val_targets),
                                  column_names=["image", "label"],
                                  num_parallel_workers=1,
                                  shuffle=True,
                                  sampler=None)

train_dataset = datapipe(trn_dataset, BATCH_SIZE)
test_dataset = datapipe(val_dataset, len(val_dataset))

images, label = (train_dataset.create_tuple_iterator()).__next__()
print(images.shape)
displayImages(images, label)

model = nn.SequentialCell(
    nn.Dense(28 * 28, 1000),
    nn.ReLU(),
    nn.Dense(1000, 10)
)


# ----------------------------------------------------------------
# summary of model
# ----------------------------------------------------------------
X = ops.randn(1, 28*28)
for blk in model:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)


loss_fn = nn.CrossEntropyLoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=LEARNING_RATE)

def train(model, dataset, loss_fn, optimizer):

    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        # 获得损失 depend用来处理操作间的依赖关系
        loss = ops.depend(loss, optimizer(grads))
        return loss

    model.set_train()
    train_epoch_losses, train_epoch_accuracies = [], []

    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        # 批量训练获得损失值
        loss = train_step(data, label)
        train_epoch_losses.append(float(loss.item()))

    train_epoch_loss = np.array(train_epoch_losses).mean()
    return train_epoch_loss


train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

for t in range(EPOCH):
    print(f"Epoch {t+1} ", '-'*80)
    loss = train(model, train_dataset, loss_fn, optimizer)
    train_losses.append(loss)
    accu, _ = accuracy(model, train_dataset, loss_fn)
    train_accuracies.append(accu)
    test_accu, test_loss = accuracy(model, test_dataset, loss_fn)
    val_losses.append(test_loss)
    val_accuracies.append(test_accu)
    print(f"Train Avg loss: {loss:>8f} Train accuracy: {(accu*100):>0.1f}% \
    Test Avg loss: {test_loss:>8f} Test accuracy: {(100*test_accu):>0.1f}%\n")


ix = 24300
plt.imshow(tr_images[ix], cmap='gray')
plt.title(fmnist_classes[tr_targets[ix]])
img = (tr_images[ix]/255.0)
print(img.shape)
img = ms.Tensor(img.reshape(-1, 28*28), dtype=ms.float32)
np_output = model(img).asnumpy()
print('np.exp(np_output)/np.sum(np.exp(np_output)): ', np.exp(np_output)/np.sum(np.exp(np_output)))


tr_targets[ix]
preds = []
for px in range(-6,6):
    img = (tr_images[ix]/255.0)
    img = img.reshape(-1, 28*28)
    img2 = np.roll(img, px, axis=1)
    img3 = ms.Tensor(img2, dtype=ms.float32)
    np_output = model(img3).asnumpy()
    pred = np.exp(np_output)/np.sum(np.exp(np_output))
    preds.append(pred)
    plt.subplot(2, 6, px + 7)
    plt.imshow(((img2 * 255).reshape(28, 28)).astype(np.uint8), cmap=plt.cm.gray)
    plt.title(fmnist_classes[pred[0].argmax()])
plt.show()

import seaborn as sns
fig, ax = plt.subplots(1,1, figsize=(12,10))
plt.title('Probability of each class for various translations')
sns.heatmap(np.array(preds).reshape(12,10), annot=True, ax=ax, fmt='.2f', xticklabels=fmnist_classes,
            yticklabels=[str(i)+str(' pixels') for i in range(-6,6)], cmap='gray')
plt.show()

exit(0)