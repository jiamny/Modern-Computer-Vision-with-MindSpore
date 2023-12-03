import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import numpy as np, pandas as pd, cv2
import selectivesearch

normalize = vision.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225], is_hwc=True)

def preprocess_image(img):
    img = normalize(img)
    img = ms.Tensor(img).permute(2,0,1)
    return img.astype(ms.float32)


class RCNNDataset:
    def __init__(self, x):
        self.x = x

    def __getitem__(self, ix):
        ix = int(ix)
        x = self.x[ix]
        return x

    def __len__(self):
        return len(self.x)


def extract_candidates(img):
    img_lbl, regions = selectivesearch.selective_search(img, scale=200, min_size=100)
    img_area = np.prod(img.shape[:2])
    candidates = []
    for r in regions:
        if r['rect'] in candidates: continue
        if r['size'] < (0.05*img_area): continue
        if r['size'] > (1*img_area): continue
        x, y, w, h = r['rect']
        candidates.append(list(r['rect']))
    return candidates


def extract_iou(boxA, boxB, epsilon=1e-5):
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    width = (x2 - x1)
    height = (y2 - y1)
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height
    area_a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    area_combined = area_a + area_b - area_overlap
    iou = area_overlap / (area_combined+epsilon)
    return iou


def getDatasets(fpaths, rois, labels, deltas, image_size):
    images, crops, bbs = [], [], []
    for ix in range(len(fpaths)):
        fpath = str(fpaths[ix])
        image = cv2.imread(fpath, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W, _ = image.shape
        sh = np.array([W,H,W,H])
        roi = rois[ix]
        bb = (np.array(roi)*sh).astype(np.uint16)
        crop = [image[y:Y,x:X] for (x,y,X,Y) in bb]
        images.append(image)
        crops.append(crop)
        bbs.append(bb)

    #print('crops: ', type(crops))

    _input, _labels, _deltas = [], [], []

    for idx in range(len(images)):
        image_crops = crops[idx]
        image_labels = labels[idx]
        image_deltas = deltas[idx]

        #print('+++++++++++++++++++++++ ', idx)
        #print(type(image_crops), ' ', image_crops[0].shape, ' ', type(image_crops[0]))
        crps = [cv2.resize(crop, (image_size, image_size)) for crop in image_crops]
        crps = [preprocess_image(crop/255.)[None] for crop in crps]
        _input.extend(crps)
        _labels.extend(image_labels)
        _deltas.extend(image_deltas)

    inputs = ms.ops.cat(_input).astype(ms.float32)
    labs = ms.Tensor(_labels, dtype=ms.int32)
    dlts = ms.Tensor(np.array(_deltas), dtype=ms.float32)
    return inputs, labs, dlts


def prepareDataset(IMAGE_ROOT, dt, batch_size, N = 500):
    FPATHS, GTBBS, CLSS, DELTAS, ROIS, IOUS = [], [], [], [], [], []

    for ix, (im, bbs, labels, fpath) in enumerate(dt):
        if(ix==N):
            break
        H, W, _ = im.shape
        candidates = extract_candidates(im)
        candidates = np.array([(x,y,x+w,y+h) for x,y,w,h in candidates])
        ious, rois, clss, deltas = [], [], [], []
        ious = np.array([[extract_iou(candidate, _bb_) for candidate in candidates] for _bb_ in bbs]).T
        for jx, candidate in enumerate(candidates):
            cx,cy,cX,cY = candidate
            candidate_ious = ious[jx]
            best_iou_at = np.argmax(candidate_ious)
            best_iou = candidate_ious[best_iou_at]
            best_bb = _x,_y,_X,_Y = bbs[best_iou_at]
            if best_iou > 0.3: clss.append(labels[best_iou_at])
            else : clss.append('background')
            delta = np.array([_x-cx, _y-cy, _X-cX, _Y-cY]) / np.array([W,H,W,H])
            deltas.append(delta)
            rois.append(candidate / np.array([W,H,W,H]))
        FPATHS.append(fpath)
        IOUS.append(ious)
        ROIS.append(rois)
        CLSS.append(clss)
        DELTAS.append(deltas)
        GTBBS.append(bbs)

    FPATHS = [f"{IMAGE_ROOT}/{f.split('/')[-1]}" for f in FPATHS]
    print(FPATHS[0])
    CLASS = []
    for cls in CLSS:
        CLASS += cls

    FPATHS, GTBBS, CLSS, DELTAS, ROIS = [item for item in [FPATHS, GTBBS, CLSS, DELTAS, ROIS]]

    # class_name对应label
    targets = pd.DataFrame(CLASS, columns=['label'])
    label2target = {l:t for t,l in enumerate(targets['label'].unique())}
    target2label = {t:l for l,t in label2target.items()}
    background_class = label2target['background']

    _CLSS = []
    for cls in CLSS:
        _cls = [label2target[c] for c in cls]
        _CLSS.append(_cls)

    n_train = 9*len(FPATHS)//10
    #print(len(FPATHS), ' ', n_train, ' ', type(FPATHS))
    #gtbbs = GTBBS[:n_train]

    train_fids = []
    for id, fpath in enumerate(FPATHS[:n_train]):
        train_fids.append(id)

    test_fids = []
    for id, fpath in enumerate(FPATHS[n_train:]):
        test_fids.append(n_train + id)

    train_ds = ds.GeneratorDataset(RCNNDataset(train_fids),
                                   column_names=["fpath"],
                                   num_parallel_workers=1,
                                   shuffle=True,
                                   sampler=None).batch(batch_size=batch_size, drop_remainder=True)

    test_ds = ds.GeneratorDataset(RCNNDataset(test_fids),
                                  column_names=["fpath"],
                                  num_parallel_workers=1,
                                  shuffle=False,
                                  sampler=None).batch(batch_size=batch_size, drop_remainder=True)
    return train_ds, test_ds, FPATHS, ROIS, _CLSS, DELTAS, label2target, target2label, n_train, test_fids



