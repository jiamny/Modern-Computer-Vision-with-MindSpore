import mindspore as ms
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import ops
import numpy as np, cv2
import matplotlib.pyplot as plt
from glob import glob

from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

from imgaug import augmenters as iaa
tfm = iaa.Sequential(iaa.Resize(28))

class XO:
    def __init__(self, folder):
        self.files = glob(folder)

    def __len__(self): return len(self.files)

    def __getitem__(self, ix):
        f = self.files[ix]
        im = tfm.augment_image(cv2.imread(f)[:,:,0])
        im = im[None]
        cl = f.split('/')[-1].split('@')[0] == 'X'
        return ms.Tensor(1 - im/255, ms.float32), ms.Tensor([cl], dtype=ms.float32)


data = XO('../data/XO/all/*')
print(len(data))

train_dataset = ds.GeneratorDataset(data, column_names=["image", "label"],
                                  num_parallel_workers=1,
                                  shuffle=True,
                                  sampler=None).batch(batch_size=64, drop_remainder=True)


R, C = 7,7
fig, ax = plt.subplots(R, C, figsize=(5,5))
for label_class, plot_row in enumerate(ax):
    for plot_cell in plot_row:
        plot_cell.grid(False); plot_cell.axis('off')
        ix = np.random.choice(1000)
        im, label = data[ix]
        plot_cell.imshow(im[0].asnumpy(), cmap='gray')
plt.tight_layout()
plt.show()

model = nn.SequentialCell(
    nn.Conv2d(1, 64, kernel_size=3),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    nn.Conv2d(64, 128, kernel_size=3),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Dense(128*7*7, 256),
    nn.ReLU(),
    nn.Dense(256, 1),
    nn.Sigmoid()
)

# ----------------------------------------------------------------
# summary of model
# ----------------------------------------------------------------
X = ops.randn(32, 1, 28, 28)
for blk in model:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)

loss_fn = nn.BCELoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=1e-3)

def accuracy(model, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        total += len(data)
        pred = pred.flatten()
        test_loss += loss_fn(pred, label.astype(ms.float32)).asnumpy()
        correct += ((pred > 0.5) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    return correct, test_loss


def train(model, dataset, loss_fn, optimizer):

    def forward_fn(data, label):
        pred = model(data)
        #pred = pred.flatten()
        loss = loss_fn(pred, label.astype(ms.float32))
        return loss, pred

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


for epoch in range(5):
    print(f"Epoch {epoch+1} ", '-'*80)
    loss = train(model, train_dataset, loss_fn, optimizer)
    print(f"Train avg loss: {loss:>8f} \n")


im, c = data[2]
plt.imshow(im[0].asnumpy())
plt.show()

first_layer = nn.SequentialCell(*list(model.cells())[:1])
intermediate_output = first_layer(im[None])[0]
n = 8
fig, ax = plt.subplots(n, n, figsize=(10,10))
for ix, axis in enumerate(ax.flat):
    axis.set_title('Filter: '+str(ix))
    axis.imshow(intermediate_output[ix].asnumpy())
plt.tight_layout()
plt.show()


print(list(model.cells()))
second_layer = nn.SequentialCell(*list(model.cells())[:4])
second_intermediate_output = second_layer(im[None])[0]
print('second_intermediate_output.shape: ', second_intermediate_output.shape)
n = 11
fig, ax = plt.subplots(n, n, figsize=(10,10))
for ix, axis in enumerate(ax.flat):
    axis.imshow(second_intermediate_output[ix].asnumpy())
    axis.set_title(str(ix))
plt.tight_layout()
plt.show()

print('im.shape: ', im.shape)
x, y = (train_dataset.create_tuple_iterator()).__next__()
x2 = x[y==0]
print('len(x2): ', len(x2))

x2 = x2.view(-1,1,28,28)
first_layer = nn.SequentialCell(*list(model.cells())[:1])
first_layer_output = first_layer(x2)
print('first_layer_output.shape: ', first_layer_output.shape)
n = 4
fig, ax = plt.subplots(n, n, figsize=(10,10))
for ix, axis in enumerate(ax.flat):
    axis.imshow(first_layer_output[ix,4,:,:].asnumpy())
    axis.set_title(str(ix))
plt.tight_layout()
plt.show()

second_layer = nn.SequentialCell(*list(model.cells())[:4])
second_intermediate_output = second_layer(x2)
print('second_intermediate_output.shape: ', second_intermediate_output.shape)
n = 4
fig, ax = plt.subplots(n, n, figsize=(10,10))
for ix, axis in enumerate(ax.flat):
    axis.imshow(second_intermediate_output[ix,34,:,:].asnumpy())
    axis.set_title(str(ix))
plt.tight_layout()
plt.show()

print("len(XO('../data/XO/all/*')): ", len(XO('../data/XO/all/*')))
custom_dl = ds.GeneratorDataset(XO('../data/XO/all/*'), column_names=["image", "label"],
                    num_parallel_workers=1,
                    shuffle=True,
                    sampler=None).batch(batch_size=2498, drop_remainder=True)

x, y = (custom_dl.create_tuple_iterator()).__next__()
x2 = x[y==0]
print('len(x2): ', len(x2))
x2 = x2.view(len(x2),1,28,28)
flatten_layer = nn.SequentialCell(*list(model.cells())[:7])
flatten_layer_output = flatten_layer(x2)
print('flatten_layer_output.shape: ', flatten_layer_output.shape)
plt.figure(figsize=(100,10))
plt.imshow(flatten_layer_output.asnumpy())
plt.show()

exit(0)