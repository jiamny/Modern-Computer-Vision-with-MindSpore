
# download ataset from: https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import copy
import numpy as np
from mindspore import context, set_seed
from mindspore.nn import Momentum
from mindspore import nn, ops
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV
from mindspore import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=0, save_graphs=False)

def create_flower_dataset(cfg, sample_num=None):
    #从目录中读取图像的源数据集。
    de_dataset = ds.ImageFolderDataset(cfg.data_path, num_samples=sample_num,
                                       class_indexing={'daisy':0,'dandelion':1,'roses':2,'sunflowers':3,'tulips':4})
    #解码前将输入图像裁剪成任意大小和宽高比。
    transform_img = CV.RandomCropDecodeResize([cfg.image_width, cfg.image_height], scale=(0.08, 1.0), ratio=(0.75, 1.333))  #改变尺寸

    #转换输入图像；形状（H, W, C）为形状（C, H, W）。
    hwc2chw_op = CV.HWC2CHW()
    #转换为给定MindSpore数据类型的Tensor操作。
    type_cast_op = C.TypeCast(mstype.float32)

    #将操作中的每个操作应用到此数据集。
    de_dataset = de_dataset.map(input_columns="image", num_parallel_workers=1, operations=transform_img)
    de_dataset = de_dataset.map(input_columns="image", operations=hwc2chw_op, num_parallel_workers=1)
    de_dataset = de_dataset.map(input_columns="image", operations=type_cast_op, num_parallel_workers=1)
    de_dataset = de_dataset.shuffle(buffer_size=cfg.data_size)
    return de_dataset


def get_flower_dataset(cfg, sample_num=None):
    '''
    读取并处理数据
    '''
    de_dataset = create_flower_dataset(cfg, sample_num=sample_num)

    #划分训练集测试集
    (de_train,de_test)=de_dataset.split([0.8,0.2])
    #设置每个批处理的行数
    #drop_remainder确定是否删除最后一个可能不完整的批（default=False）。
    #如果为True，并且如果可用于生成最后一个批的batch_size行小于batch_size行，则这些行将被删除，并且不会传播到子节点。
    de_train=de_train.batch(cfg.batch_size, drop_remainder=True)
    #重复此数据集计数次数。
    de_test=de_test.batch(cfg.batch_size, drop_remainder=True)
    return de_train, de_test


'''
变量定义
'''
cfg = edict({
    'data_path': '/media/hhj/localssd/DL_data/flower_photos',
    'data_size': 2000,
    'image_width': 224,  # 图片宽度
    'image_height': 224,  # 图片高度
    'batch_size': 4,
    'channel': 3,   # 图片通道数
    'num_class': 5,  # 分类类别
    'weight_decay':  0.0005,
    'lr': 0.01,    # 学习率
    'loss_scale': 1.0,
    'dropout_ratio': 0.5,
    'epoch_size': 200,  # 训练次数
    'sigma': 0.01,
    'momentum': 0.9
})


set_seed(1234)

(de_train, de_test) = get_flower_dataset(cfg, cfg.data_size)
batch_num = de_train.get_dataset_size()
print('batch_num：', batch_num)
print('训练数据集数量：',de_train.get_dataset_size()*cfg.batch_size) #get_dataset_size()获取批处理的大小。
print('测试数据集数量：',de_test.get_dataset_size()*cfg.batch_size)

d_test = copy.deepcopy(de_test)
data_next=d_test.create_dict_iterator(output_numpy=True).__next__()

images = data_next["image"]
labels = data_next["label"]
print(f"Image shape: {images.shape}, Label: {labels}")
classes = ['daisy','dandelion','roses','sunflowers','tulips']

plt.figure()
for i in range(cfg['batch_size']):
    plt.subplot(2, cfg['batch_size'] // 2, i+1)
    image_trans = np.transpose(images[i], (1, 2, 0)).astype(np.uint8)
    plt.title(f"{classes[labels[i]]}")
    plt.imshow(image_trans)
    plt.axis("off")
plt.show()

"""
训练
"""

step_size_train = de_train.get_dataset_size()
from vgg16 import VGG16
# 定义VGG16
vgg16 = VGG16(imgScale=cfg['image_width'] // 32, numClasses=len(classes))

#param.requires_grad = True表示所有参数都需要求梯度进行更新。
for param in vgg16.get_parameters():
    param.requires_grad = True

# 设置训练的轮数和学习率
num_epochs = cfg['epoch_size']

#基于余弦衰减函数计算学习率。学习率最小值为0.0001，最大值为0.0005，具体API见文档https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.cosine_decay_lr.html?highlight=cosine_decay_lr
lr = nn.cosine_decay_lr(min_lr=0.0001, max_lr=0.0005, total_step=step_size_train * num_epochs,
                        step_per_epoch=step_size_train, decay_epoch=num_epochs)
# 定义优化器和损失函数
# optimizer
opt = Momentum(params=vgg16.trainable_params(),
               learning_rate=ms.Tensor(lr),
               momentum=cfg.momentum,
               weight_decay=cfg.weight_decay,
               loss_scale=cfg.loss_scale)

#Adam优化器，具体可参考论文https://arxiv.org/abs/1412.6980
#opt = nn.Adam(params=vgg16.trainable_params(), learning_rate=lr)
# 交叉熵损失
loss_fn = nn.CrossEntropyLoss()

#前向传播，计算loss
def forward_fn(inputs, targets):
    logits = vgg16(inputs)
    loss = loss_fn(logits, targets)
    return loss

#计算梯度和loss
grad_fn = ops.value_and_grad(forward_fn, None, opt.parameters)

def train_step(inputs, targets):
    loss, grads = grad_fn(inputs, targets)
    loss = ops.depend(loss, opt(grads))
    return loss


# 实例化模型
model = ms.Model(vgg16, loss_fn, opt, metrics={"Accuracy": nn.Accuracy()})

# 创建迭代器
data_loader_train = de_train.create_tuple_iterator(num_epochs=num_epochs)
data_loader_val = de_test.create_tuple_iterator(num_epochs=num_epochs)

# 最佳模型存储路径
best_acc = 0
best_ckpt_dir = "./BestCheckpoint"
best_ckpt_path = "./BestCheckpoint/vgg16-flower-best.ckpt"

import os
import stat

# 开始循环训练
print("Start Training Loop ...")

for epoch in range(num_epochs):
    losses = []
    vgg16.set_train()

    # 为每轮训练读入数据

    for i, (images, labels) in enumerate(data_loader_train):
        loss = train_step(images, labels)
        if i%100 == 0 or i == step_size_train -1:
            print('Epoch: [%3d/%3d], Steps: [%3d/%3d], Train Loss: [%5.3f]'%(
                epoch+1, num_epochs, i+1, step_size_train, loss))
        losses.append(loss)

    # 每个epoch结束后，验证准确率

    acc = model.eval(de_test)['Accuracy']

    print("-" * 50)
    print("Epoch: [%3d/%3d], Average Train Loss: [%5.3f], Accuracy: [%5.3f]" % (
        epoch+1, num_epochs, sum(losses)/len(losses), acc
    ))
    print("-" * 50)

    if acc > best_acc:
        best_acc = acc
        if not os.path.exists(best_ckpt_dir):
            os.mkdir(best_ckpt_dir)
        if os.path.exists(best_ckpt_path):
            os.chmod(best_ckpt_path, stat.S_IWRITE)#取消文件的只读属性，不然删不了
            os.remove(best_ckpt_path)
        ms.save_checkpoint(vgg16, best_ckpt_path)

print("=" * 80)
print(f"End of validation the best Accuracy is: {best_acc: 5.3f}, "
      f"save the best ckpt file in {best_ckpt_path}", flush=True)


"""
验证和评估效果并且将效果可视化
"""
def visualize_model(best_ckpt_path, dataset_val):
    net = VGG16(imgScale=cfg['image_width'] // 32, numClasses=len(classes))
    # 加载模型参数
    param_dict = ms.load_checkpoint(best_ckpt_path)
    ms.load_param_into_net(net, param_dict)
    net.set_train(False)
    model = ms.Model(net)

    # 加载验证集的数据进行验证
    next(dataset_val.create_dict_iterator())
    data = next(dataset_val.create_dict_iterator())
    images = data["image"].asnumpy()
    labels = data["label"].asnumpy()
    # 预测图像类别
    output = model.predict(ms.Tensor(data['image']))
    pred = np.argmax(output.asnumpy(), axis=1)

    # 图像分类, 显示图像及图像的预测值
    plt.figure()
    for i in range(cfg['batch_size']):
        plt.subplot(2, cfg['batch_size'] // 2, i+1)
        # 若预测正确，显示为蓝色；若预测错误，显示为红色
        color = 'blue' if pred[i] == labels[i] else 'red'
        plt.title('predict:{}'.format(classes[pred[i]]), color=color)
        picture_show = np.transpose(images[i], (1, 2, 0)).astype(np.uint8)
        plt.imshow(picture_show)
        plt.axis('off')

    plt.show()

# 使用测试数据集进行验证
visualize_model(best_ckpt_path=best_ckpt_path, dataset_val=de_test)

exit(0)




