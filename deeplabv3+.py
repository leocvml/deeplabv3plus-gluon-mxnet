from mxnet import gluon, image, ndarray
from matplotlib import pyplot as plt
from mxnet.gluon import data as gdata
from os import listdir
from mxnet.gluon import nn
import mxnet as mx
from mxnet import nd


#####################################################################################
##
## DeepLab
##
####################################################################################
class stemblock(nn.HybridBlock):
    def __init__(self, filters):
        super(stemblock, self).__init__()
        self.filters = filters
        self.conv1 = nn.Conv2D(self.filters, kernel_size=3, padding=1, strides=2)
        self.bn1 = nn.BatchNorm()
        self.act1 = nn.Activation('relu')

        self.conv2 = nn.Conv2D(self.filters,kernel_size=3, padding=1, strides=1)
        self.bn2 = nn.BatchNorm()
        self.act2 = nn.Activation('relu')

        self.conv3 = nn.Conv2D(self.filters,kernel_size=3, padding=1,strides=1)

        self.pool = nn.MaxPool2D(pool_size=(2, 2), strides=2)

    def hybrid_forward(self, F, x):
        stem1 = self.act1(self.bn1(self.conv1(x)))
        stem2 = self.act2(self.bn2(self.conv2(stem1)))
        stem3 = self.pool(stem2)
        out = self.conv3(stem3)
        return out


class conv_block(nn.HybridBlock):
    def __init__(self, filters):
        super(conv_block, self).__init__()
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2D(filters, kernel_size=1),
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2D(filters, kernel_size=3, padding=1)
            )

    def hybrid_forward(self, F, x):
        return self.net(x)


class DenseBlcok(nn.HybridBlock):
    def __init__(self, num_convs, num_channels):  # layers, growth rate
        super(DenseBlcok, self).__init__()
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            for _ in range(num_convs):
                self.net.add(
                    conv_block(num_channels)
                )

    def hybrid_forward(self, F, x):
        for blk in self.net:
            Y = blk(x)
            x = F.concat(x, Y, dim=1)

        return x


class Deeplabv3(nn.HybridBlock):
    def __init__(self, growth_rate, numofcls):
        super(Deeplabv3, self).__init__()
        self.feature_extract = nn.HybridSequential()
        with self.feature_extract.name_scope():
            self.feature_extract.add(
                stemblock(256),
                DenseBlcok(6, growth_rate),
                nn.BatchNorm(),
                nn.Activation('relu')
            )
        self.conv1 = nn.HybridSequential()
        with self.conv1.name_scope():
            self.conv1.add(
                nn.Conv2D(32, kernel_size=1, strides=2),
                nn.BatchNorm(),
                nn.Activation('relu')
            )

        self.conv3r6 = nn.HybridSequential()
        with self.conv3r6.name_scope():
            self.conv3r6.add(
                nn.Conv2D(32, kernel_size=3, strides=2, padding=6, dilation=6),
                nn.BatchNorm(),
                nn.Activation('relu')
            )
        self.conv3r12 = nn.HybridSequential()
        with self.conv3r12.name_scope():
            self.conv3r12.add(
                nn.Conv2D(32, kernel_size=3, strides=2, padding=12, dilation=12),
                nn.BatchNorm(),
                nn.Activation('relu')
            )
        self.conv3r18 = nn.HybridSequential()
        with self.conv3r18.name_scope():
            self.conv3r18.add(
                nn.Conv2D(32, kernel_size=3, strides=2, padding=18, dilation=18),
                nn.BatchNorm(),
                nn.Activation('relu')
            )

        self.maxpool = nn.MaxPool2D(pool_size=2, strides=2)

        self.concatconv1 = nn.HybridSequential()
        with self.concatconv1.name_scope():
            self.concatconv1.add(

                nn.Conv2D(256, kernel_size=1),
                nn.BatchNorm(),
                nn.Activation('relu')
            )

        self.feconv1 = nn.HybridSequential()
        with self.feconv1.name_scope():
            self.feconv1.add(
                nn.Conv2D(256, kernel_size=1),
                nn.BatchNorm(),
                nn.Activation('relu')
            )
        self.transUp = nn.HybridSequential()
        with self.transUp.name_scope():
            self.transUp.add(
                nn.Conv2DTranspose(256, kernel_size=4, padding=1, strides=2),
                nn.BatchNorm(),
                nn.Activation('relu')
            )
        self.decodeConv3 = nn.HybridSequential()
        with self.decodeConv3.name_scope():
            self.decodeConv3.add(
                nn.Conv2D(256, kernel_size=3, padding=1, strides=1),
                nn.BatchNorm(),
                nn.Activation('relu')
            )
        self.Up4 = nn.HybridSequential()
        with self.Up4.name_scope():
            self.Up4.add(
                nn.Conv2DTranspose(256, kernel_size=4, padding=1, strides=2),
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2DTranspose(numofcls, kernel_size=4, padding=1, strides=2)
            )

    def hybrid_forward(self, F, x):
        out = self.feature_extract(x)

        conv1out = self.conv1(out)
        conv3r6out = self.conv3r6(out)
        conv3r12out = self.conv3r12(out)
        conv3r18out = self.conv3r18(out)
        maxpoolout = self.maxpool(out)

        second_out = ndarray.concat(conv1out, conv3r6out, conv3r12out, conv3r18out, maxpoolout, dim=1)
        encoder_out = self.concatconv1(second_out)
        encoderUp = self.transUp(encoder_out)
        feconv1out = self.feconv1(out)

        combine_out = ndarray.concat(encoderUp, feconv1out, dim=1)
        output = self.decodeConv3(combine_out)
        output = self.Up4(output)

        return output


class SegDataset(gluon.data.Dataset):
    def __init__(self, root, resize, colormap=None, classes=None):
        self.root = root
        self.resize = resize
        self.colormap = colormap
        self.classes = classes
        self.colormap2label = None
        self.load_images()

    def label_indices(self, img):
        if self.colormap2label is None:
            self.colormap2label = nd.zeros(256 ** 3)

            for i, cm in enumerate(self.colormap):
                self.colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        data = img.astype('int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return self.colormap2label[idx]

    def read_images(self, root):

        dataroot = root + 'data/'  # left_frames   #data
        labelroot = root + 'label/'  # labels   #label
        DataNamelist = [f for f in listdir(dataroot)]
        labelNamelist = [f for f in listdir(labelroot)]
        DataNamelist = sorted(DataNamelist)
        labelNamelist = sorted(labelNamelist)

        if len(DataNamelist) != len(labelNamelist):
            raise ValueError('number of your data is different from label')
        else:
            data, label = [None] * len(DataNamelist), [None] * len(labelNamelist)

            for i, name in enumerate(DataNamelist):
                data[i] = image.imread(dataroot + name)

            for i, name in enumerate(labelNamelist):
                label[i] = image.imread(labelroot + name)

            return data, label

    def load_images(self):
        data, label = self.read_images(root=self.root)
        self.data = [self.normalize_image(im) for im in data]
        if self.colormap is None:
            self.label = [self.normalize_image(im) for im in label]

        if self.colormap != None:
            self.label = label

        print('read ' + str(len(self.data)) + ' examples')

    def normalize_image(self, data):
        return data.astype('float32') / 255

    def __getitem__(self, item):
        if self.colormap is None:
            data = image.imresize(self.data[item], self.resize[0], self.resize[1])
            label = image.imresize(self.label[item], self.resize[0], self.resize[1])

            return data.transpose((2, 0, 1)), label.transpose((2, 0, 1))
        if self.colormap != None:
            data = image.imresize(self.data[item], self.resize[0], self.resize[1])
            label = image.imresize(self.label[item], self.resize[0], self.resize[1])

            return data.transpose((2, 0, 1)), self.label_indices(label)

    def __len__(self):
        return len(self.data)


voc_colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128],
                [128, 128, 128], [64, 0, 0], [192, 0, 0],
                [64, 128, 0], [192, 128, 0], [64, 0, 128],
                [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0],
                [128, 192, 0], [0, 64, 128]]

classes = ['background', 'aeroplane', 'bicycle', 'bird',
           'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'potted plant', 'sheep', 'sofa',
           'train', 'tv/monitor']
numcls = len(voc_colormap)


def LoadDataset(dir, batchsize, output_shape, colormap=None, classes=None):
    dataset = SegDataset(dir, output_shape, colormap, classes)

    data_iter = gdata.DataLoader(dataset, batchsize, shuffle=True)

    return data_iter


train_dir = 'VOC_less/'
test_dir = 'VOC_less/'
ctx = mx.gpu()
batch_size = 3
resize = (480, 320)
train_iter = LoadDataset(train_dir, batch_size, resize, voc_colormap,
                         classes)  # default is for 2 class if you want to multiclass
test_iter = LoadDataset(test_dir, batch_size, resize, voc_colormap, classes)

for d, l in train_iter:
    break

print(d.shape)
print(l.shape)

#####################################################################
###
###  Net training
###
###
#####################################################################
net = nn.HybridSequential()
with net.name_scope():
    net.add(
        Deeplabv3(growth_rate=12, numofcls=numcls)  # 12
    )

net.initialize()

softmax_CE = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
net.collect_params().reset_ctx(ctx=ctx)
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.001})
filename = 'VOC_less.params'
net.load_params(filename, ctx=ctx)
import time
from mxnet import autograd

for epoch in range(0):

    tic = time.time()
    for i, (d, l) in enumerate(train_iter):
        x = d.as_in_context(ctx)
        y = l.as_in_context(ctx)

        with autograd.record():
            predict = net(x)

            loss = softmax_CE(predict, y)

        loss.backward()
        trainer.step(x.shape[0])
        # update metrics

    print('Epoch %2d,loss %.5f, time %.1f sec' % (
        epoch, mx.ndarray.mean(loss).asscalar(), time.time() - tic))
    net.save_params(filename)

net.save_params(filename)
#################################################################################
######
#####   Netweork inference
#####
###############################################################################
import numpy as np


def predict2img(predict):
    colormap = ndarray.array(train_iter._dataset.colormap, ctx=mx.gpu(), dtype='uint8')  # voc_colormap
    # colormap = np.array(train_iter._dataset.colormap).astype('uint8')
    target = predict.asnumpy()
    label = colormap[target[:, :, :]]
    return label.asnumpy()


from skimage import io

for d, l in train_iter:
    break
d = d.as_in_context(ctx)
target = net(d)
target = ndarray.argmax(target, axis=1)
predict = predict2img(target)

######################################################################
####    show image
######################################################################
d = ndarray.transpose(d, (0, 2, 3, 1))

label = predict2img(l)

figsize = (10, 10)
_, axes = plt.subplots(3, d.shape[0], figsize=figsize)

for i in range(d.shape[0]):
    axes[0][i].imshow(d[i, :, :, :].asnumpy())
    axes[1][i].imshow(predict[i, :, :, :])
    axes[2][i].imshow(label[i, :, :, :])

plt.show()

