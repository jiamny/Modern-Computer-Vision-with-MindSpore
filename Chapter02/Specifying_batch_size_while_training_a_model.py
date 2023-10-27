import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net
import mindspore.context as context
import mindspore.dataset as ds
from matplotlib import pyplot as plt
import numpy as np

ms.set_seed(10)
context.set_context(device_target="GPU")

X = Tensor([[1,2],[3,4],[5,6],[7,8]], dtype=ms.float32)
Y = Tensor([[3],[7],[11],[15]], dtype=ms.float32)

class MyDataset:
    def __init__(self,x,y):
        self.x = x.copy().asnumpy()
        self.y = y.copy().asnumpy()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]


# ---------------------------------------------------------------------------------
# Specifying batch size:  batch(2, drop_remainder=False)
# ---------------------------------------------------------------------------------
dataset = MyDataset(X, Y)
dataloader = ds.GeneratorDataset(dataset, column_names=["data", "label"],
                                 shuffle=True).batch(2, drop_remainder=False)
model = nn.SequentialCell(
    nn.Dense(2, 8),
    nn.ReLU(),
    nn.Dense(8, 1)
)

print('model:\n', model)

loss_func = nn.MSELoss()

def forward(X, Y):
    logits = model(X)
    print('logits: ', logits)
    loss = loss_func(logits, Y)
    return loss, logits


opt = nn.SGD(model.trainable_params(), learning_rate = 0.001)
grad_fn = ms.value_and_grad(forward, None, opt.parameters, has_aux=False)

print('-------------------------- Train -----------------------')
iterator = dataloader.create_tuple_iterator()
import time
loss_history = []

model.set_train()

start = time.time()
for _ in range(50):
    epoch_loss = []
    for batch in iterator:
        x, y = batch
        (loss, _), grads = grad_fn(x, y)
        # 获得损失 depend用来处理操作间的依赖关系
        loss = ms.ops.depend(loss, opt(grads))
        epoch_loss.append(float(loss.item()))
    epoch_loss = np.array(epoch_loss).mean()
    loss_history.append(epoch_loss)

end = time.time()
print('Training takes time: ', end - start)

plt.plot(loss_history)
plt.title('Loss variation over increasing epochs')
plt.xlabel('epochs')
plt.ylabel('loss value')
plt.show()

# Predictions
val = Tensor([[10,11]], dtype=ms.float32)
print('--------------------- Predictions ----------------------')
print('model(val):\n', model(val))
print('val.sum(-1):\n', val.sum(-1))
