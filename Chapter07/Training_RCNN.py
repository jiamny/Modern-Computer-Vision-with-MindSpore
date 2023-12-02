import mindspore as ms
import numpy as np, pandas as pd, cv2
from mindspore.ops import composite as C
from mindspore import context, set_seed, ops, nn, load_checkpoint, load_param_into_net
import sys, time
import progressbar
sys.path.insert(0, '../utils')
from plotResults import displayBboxImage
import matplotlib.pyplot as plt

from PrepareDataset import prepareDataset, getDatasets, extract_candidates, preprocess_image
from vgg16 import VGG16

context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=0, save_graphs=False)

# ----------------------------------------
# 加载数据集
# ----------------------------------------
batch_size = 2                              # 批量大小
image_size = 224                            # 训练图像空间大小
n_epochs = 20                               # 训练周期数
lr = 0.001                                  # 学习率
num_classes = 3                             # 输出通道数大小Flower分类数5

slt = True
# 数据集目录路径
IMAGE_ROOT = '/media/hhj/localssd/DL_data/open-images-bus-trucks/images/images'
DF_RAW = pd.read_csv('/media/hhj/localssd/DL_data/open-images-bus-trucks/df.csv')
if slt:
    print(DF_RAW.shape)
    uni_ids = DF_RAW['ImageID'].unique()
    s_ids = []
    for u in uni_ids:
        if np.random.rand(1) > 0.5:
            s_ids.append(u)
    print(len(s_ids))
    DF_RAW = DF_RAW.loc[DF_RAW['ImageID'].isin(s_ids)]
    DF_RAW = DF_RAW.sort_values(by='ImageID')
    print(DF_RAW.shape)


print(len(DF_RAW['ImageID'].unique()))

class OpenImages():
    def __init__(self, df, image_folder=IMAGE_ROOT):
        self.root = image_folder
        self.df = df
        self.unique_images = df['ImageID'].unique()

    def __len__(self): return len(self.unique_images)

    def __getitem__(self, ix):
        image_id = self.unique_images[ix]
        image_path = f'{self.root}/{image_id}.jpg'
        image = cv2.imread(image_path, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # conver BGR to RGB
        h, w, _ = image.shape
        df = self.df.copy()
        df = df[df['ImageID'] == image_id]
        boxes = df['XMin,YMin,XMax,YMax'.split(',')].values
        boxes = (boxes * np.array([w,h,w,h])).astype(np.int32).tolist()
        classes = df['LabelName'].values.tolist()
        return image, boxes, classes, image_path


dt = OpenImages(df=DF_RAW)
im, bbs, clss, _ = dt[9]
displayBboxImage(im, bbs, clss)

train_ds, test_ds, FPATHS, ROIS, _CLSS, DELTAS, label2target, target2label, n_train, test_fids = prepareDataset(IMAGE_ROOT, dt, batch_size, N=300)
background_class = label2target['background']


def collate_fn(fids, dt_fpaths, dt_rois, dt_clss, dt_deltas):
    #fpaths = [dt_fpaths[int(fids[0][i].item())] for i in range(len(fids[0]))]
    fids = fids['fpath']
    fpaths = [dt_fpaths[int(fids[i].item())] for i in range(fids.shape[0])]
    rois = [dt_rois[int(fids[i].item())] for i in range(fids.shape[0])]
    labels = [dt_clss[int(fids[i].item())] for i in range(fids.shape[0])]
    deltas = [dt_deltas[int(fids[i].item())] for i in range(fids.shape[0])]
    return getDatasets(fpaths, rois, labels, deltas, image_size)


def where_not_zero(mx):
    mp = []
    for i in range(len(mx)):
        if mx[i] != 0:
            mp.append(i)
    return ms.Tensor(mp, dtype=ms.int32)


backbone = VGG16()
X = ms.Tensor(np.random.rand(1, 3, 224, 224), ms.float32)
for blk in backbone.layers:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)

print('*'*80)

def decode(_y):
    _, preds = ops.max(_y, -1)
    return preds


class RCNN(nn.Cell):
    def __init__(self):
        super().__init__()
        feature_dim = 25088
        self.backbone = backbone
        self.cls_score = nn.Dense(feature_dim, len(label2target))
        self.bbox = nn.SequentialCell(
            nn.Dense(feature_dim, 512),
            nn.ReLU(),
            nn.Dense(512, 4),
            nn.Tanh(),
        )

    def construct(self, x):
        x = self.backbone(x)
        clss = self.cls_score(x)
        bbs = self.bbox(x)

        return clss, bbs


class LossNet(nn.Cell):
    """MaskRcnn loss method"""
    def construct(self, x1, x2, x3):
        lmb = 10.0
        return x1 + lmb * x2


# ----------------------------------------------------------------
# train model
# ----------------------------------------------------------------

rcnn = RCNN()
for param in rcnn.get_parameters():
    param.requires_grad = True

rcnn.set_train()
cel = nn.CrossEntropyLoss()
sl1 = nn.L1Loss()

loss_fn = LossNet()
opt = nn.SGD(rcnn.trainable_params(), learning_rate=lr)

#计算梯度和loss
grad_fn = ops.value_and_grad(loss_fn, None, opt.parameters, has_aux=False)


for epoch in range(n_epochs):

    tr_loss, tr_acc = [], []
    sys.stdout.flush()
    # train model
    rcnn.set_train(True)
    _n = train_ds.get_dataset_size()
    bar = progressbar.ProgressBar(maxval=_n).start()

    for ix, fids in enumerate(train_ds.create_dict_iterator(num_epochs=n_epochs)):
        #print('fids: ', fids)
        inputs, clss, deltas = collate_fn(fids, FPATHS, ROIS, _CLSS, DELTAS)
        cls_score, _deltas = rcnn(inputs)

        detection_loss = cel(cls_score, clss)
        ixs = where_not_zero(clss)      # ms.ops.where(labels != 0)
        _deltas = _deltas[ixs]
        deltas = deltas[ixs]

        if len(ixs) > 0:
            regression_loss = sl1(_deltas, deltas)
        else:
            regression_loss = 0

        accs = 1.0 * (clss == decode(cls_score)).astype(ms.int32).sum()/len(clss)

        (loss), grads = grad_fn(detection_loss, regression_loss, accs)
        loss = ops.depend(loss, opt(grads))
        tr_loss.append(float(loss.item()))
        tr_acc.append(float(accs.item()))
        bar.update(ix)
    bar.finish()
    print('\nTraining epoch: ', (epoch+1), ' loss: ', np.mean(tr_loss), ' acc: ', np.mean(tr_acc))
    time.sleep(1)
    
    sys.stdout.flush()
    # validate model
    ts_loss, ts_acc = [], []
    rcnn.set_train(False)
    _n = test_ds.get_dataset_size()
    bar = progressbar.ProgressBar(maxval=_n).start()
    for ix, fids in enumerate(test_ds.create_dict_iterator(num_epochs=n_epochs)):

        inputs, clss, deltas = collate_fn(fids, FPATHS, ROIS, _CLSS, DELTAS)
        cls_score, _deltas = rcnn(inputs)

        detection_loss = cel(cls_score, clss)
        ixs = where_not_zero(clss)      # ms.ops.where(labels != 0)
        _deltas = _deltas[ixs]
        deltas = deltas[ixs]

        if len(ixs) > 0:
            regression_loss = sl1(_deltas, deltas)
        else:
            regression_loss = 0

        loss = detection_loss + 10.0 * regression_loss
        accs = 1.0 * (clss == decode(cls_score)).astype(ms.int32).sum()/len(clss)
        ts_loss.append(float(loss.item()))
        ts_acc.append(float(accs.item()))
        bar.update(ix)
    bar.finish()
    print('\nValidation epoch: ', (epoch+1), ' loss: ', np.mean(ts_loss), ' acc: ', np.mean(ts_acc))
    time.sleep(1)

'''
ms.save_checkpoint(rcnn, "./rcnn.ckpt")
param_dict = load_checkpoint("./rcnn.ckpt")
load_param_into_net(rcnn, param_dict)
'''

# Plotting training and validation metrics
def tst_predictions(filename, show_output=True):
    img = cv2.imread(filename, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #[...,::-1]
    #img = np.array( img )
    candidates = extract_candidates(img)
    candidates = [(x,y,x+w,y+h) for x,y,w,h in candidates]
    input = []
    for candidate in candidates:
        x,y,X,Y = candidate
        crop = cv2.resize(img[y:Y,x:X], (224,224))
        input.append(preprocess_image(crop/255.)[None])
    input = ops.cat(input)

    rcnn.set_train(False)
    probs, deltas = rcnn(input)
    probs = ops.softmax(probs, -1)
    confs, clss = ops.max(probs, -1)
    candidates = np.array(candidates)
    confs, clss, probs, deltas = [tensor.asnumpy() for tensor in [confs, clss, probs, deltas]]

    ixs = clss != background_class
    confs, clss, probs, deltas, candidates = [tensor[ixs] for tensor in [confs, clss, probs, deltas, candidates]]
    bbs = (candidates + deltas).astype(np.uint16)
    box_with_score = np.column_stack((bbs.astype(np.float32), confs))
    box_with_score_m = ms.Tensor(box_with_score)
    iou_threshold = 0.05
    output_boxes, output_idx, ixs = ops.NMSWithMask(iou_threshold)(box_with_score_m)
    ixs = ixs.asnumpy()
    confs, clss, probs, deltas, candidates, bbs = [tensor[ixs] for tensor in [confs, clss, probs, deltas, candidates, bbs]]

    if len(ixs) == 1:
        confs, clss, probs, deltas, candidates, bbs = [tensor[None] for tensor in [confs, clss, probs, deltas, candidates, bbs]]

    if len(confs) == 0 and not show_output:
        return (0,0,224,224), 'background', 0

    if len(confs) > 0:
        best_pred = np.argmax(confs)
        best_conf = np.max(confs)
        best_bb = bbs[best_pred]
        x,y,X,Y = best_bb

    print("bbs: ", bbs, '\n best_bbs: ', best_bb)

    _, ax = plt.subplots(1, 2, figsize=(12,6))
    ax[0].imshow(img)
    ax[0].grid(False)
    ax[0].set_title('Original image')
    if len(confs) == 0:
        ax[1].imshow(img)
        ax[1].set_title('No objects')
        plt.show()
        return
    ax[1].set_title('predicted bounding box and class') #target2label[clss[best_pred]])
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.3
    thinkness = 1
    for i, rect in enumerate(bbs):
        # draw rectangle for region proposal till numShowRects
        print('rect: ', rect, ' predicted class: ', target2label[clss[i]])
        xmin, ymin, xmax, ymax = rect
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), thinkness, cv2.LINE_AA)

        (w, h), baseline = cv2.getTextSize( target2label[clss[i]], fontFace, fontScale, thinkness )

        cv2.rectangle(img, (xmin, ymin - h - baseline - thinkness),(xmin + w, ymin), (0, 0, 255), -1)
        cv2.putText(img, target2label[clss[i]], (xmin, ymin - baseline),
                fontFace, fontScale, (255, 255, 255), thinkness, cv2.LINE_AA)
    ax[1].imshow(img)
    #plt.suptitle('predicted bounding box and class')
    plt.show()
    return (x,y,X,Y),target2label[clss[best_pred]],best_conf
    

fpath = FPATHS[int(test_fids[5])]
print('fapath: ', fpath)
tst_predictions(fpath)

exit(0)