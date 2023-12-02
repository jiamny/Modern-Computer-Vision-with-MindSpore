import numpy as np, cv2, sys
import matplotlib.pyplot as plt

def displayImages(images, label):
    # 显示10张图片
    images = (images*255).asnumpy().astype(np.uint8)

    for i in range(images.shape[0]):
        image = images[i].reshape((28,28))

        if i >= 10:
            break
        plt.subplot(2, 5, i + 1)
        plt.imshow(image.squeeze(), cmap=plt.cm.gray)
        plt.title(label[i])
    plt.show()


def showLossAndAccuracy(train_losses, train_accuracies, val_losses, val_accuracies, EPOCH, tlt):
    epochs = np.arange(EPOCH)+1

    plt.figure(figsize=(10, 12))
    plt.subplot(211)
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.title('Training and validation loss with ' + tlt)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(False)

    plt.subplot(212)
    plt.plot(epochs, train_accuracies, 'b', label='Training accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy with ' + tlt)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    yl = [x for x in plt.gca().get_yticks()]
    plt.gca().set_yticks(yl)
    plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in yl])
    plt.legend()
    plt.grid(False)
    plt.show()


def showTrainLossAndAccuracy(train_losses, train_accuracies, EPOCH, tlt):
    epochs = np.arange(EPOCH)+1
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.title('Loss over epochs ' + tlt)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.legend()

    plt.subplot(122)
    plt.title('Accuracy over epochs ' + tlt)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    yl = [x for x in plt.gca().get_yticks()]
    plt.gca().set_yticks(yl)
    plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in yl])
    plt.legend()
    plt.show()

def showWeightDistribution(model):
    plt.figure(figsize=(16, 16))
    for ix, par in enumerate(model.trainable_params()):
        if(ix==0):
            plt.subplot(2, 2, ix + 1)
            plt.hist(par.asnumpy().flatten())
            plt.title('Weights conencting input to hidden layer')
        elif(ix ==1):
            plt.subplot(2, 2, ix + 1)
            plt.hist(par.asnumpy().flatten())
            plt.title('Biases of hidden layer')
        elif(ix==2):
            plt.subplot(2, 2, ix + 1)
            plt.hist(par.asnumpy().flatten())
            plt.title('Weights conencting hidden to output layer')
        elif(ix ==3):
            plt.subplot(2, 2, ix + 1)
            plt.hist(par.asnumpy().flatten())
            plt.title('Biases of output layer')
    plt.show()


def displayBboxImage(im, bbs, clss):
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.3
    thinkness = 1
    # itereate over all the bbox region
    for i, rect in enumerate(bbs):
        # draw rectangle for region proposal till numShowRects
        print(rect)
        xmin, ymin, xmax, ymax = rect
        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (0, 255, 0), thinkness, cv2.LINE_AA)


        (w, h), baseline = cv2.getTextSize( clss[0], fontFace, fontScale, thinkness )

        cv2.rectangle(im, (xmin, ymin - h - baseline - thinkness),(xmin + w, ymin), (0, 0, 255), -1)
        cv2.putText(im, clss[0], (xmin, ymin - baseline),
               fontFace, fontScale, (255, 255, 255), thinkness, cv2.LINE_AA)

    plt.figure(figsize=(8, 5))
    plt.imshow(im)
    plt.title('Object detected')
    plt.show()

