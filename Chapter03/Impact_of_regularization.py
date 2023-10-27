
import mindspore
from mindspore import nn
from mindspore import ops
from mindspore.dataset import vision
from mindspore.dataset import transforms
from mindspore.dataset import FashionMnistDataset
import numpy as np
import sys
sys.path.insert(0, '../utils')
from plotResults import displayImages, showLossAndAccuracy, showWeightDistribution

from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=0)


BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCH = 10

trn_dataset = FashionMnistDataset(dataset_dir='/media/hhj/localssd/DL_data/fashion_MNIST/FashionMNIST/',
                                  usage='train',
                                  num_parallel_workers = 1,
                                  shuffle=True,
                                  num_samples=None)
tst_dataset = FashionMnistDataset(dataset_dir='/media/hhj/localssd/DL_data/fashion_MNIST/FashionMNIST/',
                                  usage='test',
                                  num_parallel_workers = 1,
                                  shuffle=False,
                                  num_samples=None)


def datapipe(dataset, batch_size):
    image_transforms = [
        # 基于给定的缩放和平移因子调整图像的像素大小。输出图像的像素大小为：output = image * rescale + shift。
        # 此处rescale取1.0 / 255.0，shift取0
        vision.Rescale(1.0 / 255.0, 0),
        # 将输入图像的shape从 <H, W, C> 转换为 <C, H, W>
        vision.HWC2CHW()
    ]

    label_transform = transforms.TypeCast(mindspore.int32)

    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')

    # 将数据集中连续 batch_size 条数据组合为一个批数据
    dataset = dataset.batch(batch_size)
    return dataset


train_dataset = datapipe(trn_dataset, BATCH_SIZE)
test_dataset = datapipe(tst_dataset, len(tst_dataset))

#images, label = train_dataset.create_tuple_iterator().__next__()
#displayImages(images, label)

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

# -----------------------------------------------------------------
# Model no regularization
# -----------------------------------------------------------------

model1 = nn.SequentialCell(
    nn.Flatten(),
    nn.Dense(28 * 28, 1000),
    nn.ReLU(),
    nn.Dense(1000, 10)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = nn.Adam(model1.trainable_params(), learning_rate=LEARNING_RATE)

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

print('*'*40, ' Model no regularization ', '*'*40)
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

for t in range(EPOCH):
    print(f"Epoch {t+1} ", '-'*80)
    loss = train(model1, train_dataset, loss_fn, optimizer)
    train_losses.append(loss)
    accu, _ = accuracy(model1, train_dataset, loss_fn)
    train_accuracies.append(accu)
    test_accu, test_loss = accuracy(model1, test_dataset, loss_fn)
    val_losses.append(test_loss)
    val_accuracies.append(test_accu)
    print(f"Train Avg loss: {loss:>8f} Train accuracy: {(accu*100):>0.1f}% \
    Test Avg loss: {test_loss:>8f} Test accuracy: {(100*test_accu):>0.1f}%\n")


showLossAndAccuracy(train_losses, train_accuracies, val_losses, val_accuracies, EPOCH, 'no regularization')

# -----------------------------------------------------------------
# Model with Regularization - 1e-4
# -----------------------------------------------------------------

model2 = nn.SequentialCell(
    nn.Flatten(),
    nn.Dense(28 * 28, 1000),
    nn.ReLU(),
    nn.Dense(1000, 10)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = nn.Adam(model2.trainable_params(), learning_rate=LEARNING_RATE)

def train(model, dataset, loss_fn, optimizer):

    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)

        # --------------------------------------------------------------
        # Regularization - 1e-4
        # --------------------------------------------------------------
        l1_regularization = 0
        for _, param in enumerate(model.trainable_params()):
            l1_regularization += float(ops.norm(param,1).item())
        loss += 0.0001*l1_regularization

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


print('*'*40, ' Model with regularization - 1e-4 ', '*'*40)
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

for t in range(EPOCH):
    print(f"Epoch {t+1} ", '-'*80)
    loss = train(model2, train_dataset, loss_fn, optimizer)
    train_losses.append(loss)
    accu, _ = accuracy(model2, train_dataset, loss_fn)
    train_accuracies.append(accu)
    test_accu, test_loss = accuracy(model2, test_dataset, loss_fn)
    val_losses.append(test_loss)
    val_accuracies.append(test_accu)
    print(f"Train Avg loss: {loss:>8f} Train accuracy: {(accu*100):>0.1f}% \
    Test Avg loss: {test_loss:>8f} Test accuracy: {(100*test_accu):>0.1f}%\n")


showLossAndAccuracy(train_losses, train_accuracies, val_losses, val_accuracies, EPOCH, 'regularization 1e-4')

# -----------------------------------------------------------------
# Model with regularization 1e-2
# -----------------------------------------------------------------

model3 = nn.SequentialCell(
    nn.Flatten(),
    nn.Dense(28 * 28, 1000),
    nn.ReLU(),
    nn.Dense(1000, 10)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = nn.Adam(model3.trainable_params(), learning_rate=LEARNING_RATE)

def train(model, dataset, loss_fn, optimizer):

    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)

        # --------------------------------------------------------------
        # Regularization - 1e-2
        # --------------------------------------------------------------
        l1_regularization = 0
        for _, param in enumerate(model.trainable_params()):
            l1_regularization += float(ops.norm(param,1).item())
        loss += 0.01*l1_regularization

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


print('*'*40, ' Model with regularization - 1e-2 ', '*'*40)
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

for t in range(EPOCH):
    print(f"Epoch {t+1} ", '-'*80)
    loss = train(model3, train_dataset, loss_fn, optimizer)
    train_losses.append(loss)
    accu, _ = accuracy(model3, train_dataset, loss_fn)
    train_accuracies.append(accu)
    test_accu, test_loss = accuracy(model3, test_dataset, loss_fn)
    val_losses.append(test_loss)
    val_accuracies.append(test_accu)
    print(f"Train Avg loss: {loss:>8f} Train accuracy: {(accu*100):>0.1f}% \
    Test Avg loss: {test_loss:>8f} Test accuracy: {(100*test_accu):>0.1f}%\n")


showLossAndAccuracy(train_losses, train_accuracies, val_losses, val_accuracies, EPOCH, 'regularization 1e-2')
showWeightDistribution(model1)
showWeightDistribution(model2)
showWeightDistribution(model3)

exit(0)