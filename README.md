# deeplabv3plus-gluon-mxnet
this repo is guide to segmentation using deeplabv3plus as sample code

# deeplabv3plus_gluon  #

this repo is want to guide to Semantic Segmentation with Deep Learning using gluon



this repo attemps to reproduce [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611) use gluon reimplementation of DeepLabV3+ , **but still  different in current version**

**I use DenseNet as backbone instead of DCNN layer(Atrous Conv)**

# tutorials #
**generate dataloader for segmentaion model**
```
in the segmentaion task, we want to assign 
each pixel in the image an object class.
2 case of segmentaion task

1. binary case (front / background)
2. multiple case (many object in one image)

```
![](https://github.com/leocvml/deeplabv3plus-gluon-mxnet/blob/master/hackmdimg/vocb.jpg)
![](https://github.com/leocvml/deeplabv3plus-gluon-mxnet/blob/master/hackmdimg/segb.PNG)

## quick start ##
```python
1. clone( download)
2. execution deeplabv3+  ( training on VOC_less just 7 image)
3. show our training result ( you can change code in line 351)

```

**class introduction**


***label_indices:*** 
for mutiple class we want formulate segmentation task as classfication problem and use lookup table for any class

***read_images:***
read dataset from our folder

***load image***
first we should check which label format is our expect(binary or multiple class)
and normalize our training data

***normalize_image***
img = img / 255

***__getitem__***
we can use any data augmentation method in this function like random crop or resize ,in this sample we just use resize


```python 
class SegDataset(gluon.data.Dataset):
    def __init__(self,root,resize,colormap=None,classes=None):
        self.root = root
        self.resize = resize
        self.colormap = colormap
        self.classes = classes
        self.colormap2label = None
        self.load_images()

    def label_indices(self,img):  
        if self.colormap2label is None:
            self.colormap2label = nd.zeros(256**3)

            for i, cm in enumerate(self.colormap):

                self.colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        data = img.astype('int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return self.colormap2label[idx]

    def read_images(self,root):

        dataroot = root + 'data/'  # left_frames   #data
        labelroot = root + 'label/'  # labels   #label
        DataNamelist = [f for f in listdir(dataroot)]
        labelNamelist = [f for f in listdir(labelroot)]

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
    def normalize_image(self,data):
        return data.astype('float32') / 255

    def __getitem__(self, item):
        if self.colormap is None:
            data = image.imresize(self.data[item], self.resize[0], self.resize[1])
            label = image.imresize(self.label[item], self.resize[0], self.resize[1])

            return data.transpose((2, 0, 1)), label.transpose((2,0,1))
        if self.colormap != None:
            data = image.imresize(self.data[item], self.resize[0], self.resize[1])
            label = image.imresize(self.label[item], self.resize[0], self.resize[1])

            return data.transpose((2, 0, 1)), self.label_indices(label)


    def __len__(self):
        return len(self.data)

```

## Dataloader ##
```python
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
                    
def LoadDataset(dir, batchsize, output_shape, colormap = None, classes=None):
# select your dataset format (colormap, classes optional)
    dataset = SegDataset(dir, output_shape,colormap,classes)
    data_iter = gdata.DataLoader(dataset, batchsize, shuffle=False)

    return data_iter

dir = 'VOC_less/'
ctx = mx.gpu()
batch_size = 3
resize = (480, 320)
train_iter = LoadDataset(dir, batch_size, resize, voc_colormap, classes)   # default is for 2 class if you want to multiclass


```
## network arch ##
**follow this architecture but use densenet instead of DCNN layer**
![](https://github.com/leocvml/deeplabv3plus-gluon-mxnet/blob/master/hackmdimg/deeplabv3arch.PNG)

```python

class Deeplabv3(nn.HybridBlock):
    def __init__(self,growth_rate,numofcls):
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
                nn.Conv2D(32,kernel_size=3,strides=2,padding=18,dilation=18),
                nn.BatchNorm(),
                nn.Activation('relu')
            )

        self.maxpool = nn.MaxPool2D(pool_size=2,strides=2)

        self.concatconv1 = nn.HybridSequential()
        with self.concatconv1.name_scope():
            self.concatconv1.add(

                nn.Conv2D(256,kernel_size=1),
                nn.BatchNorm(),
                nn.Activation('relu')
            )

        self.feconv1 = nn.HybridSequential()
        with self.feconv1.name_scope():
            self.feconv1.add(
                nn.Conv2D(256,kernel_size=1),
                nn.BatchNorm(),
                nn.Activation('relu')
            )
        self.transUp = nn.HybridSequential()
        with self.transUp.name_scope():
            self.transUp.add(
                nn.Conv2DTranspose(256,kernel_size=4,padding=1,strides=2),
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
                    nn.Conv2DTranspose(numofcls,kernel_size=4,padding=1,strides=2)
                             )


    def hybrid_forward(self, F, x):
        out = self.feature_extract(x)

        conv1out = self.conv1(out)
        conv3r6out = self.conv3r6(out)
        conv3r12out = self.conv3r12(out)
        conv3r18out = self.conv3r18(out)
        maxpoolout = self.maxpool(out)

        second_out = ndarray.concat(conv1out,conv3r6out,conv3r12out,conv3r18out,maxpoolout, dim = 1)
        encoder_out = self.concatconv1(second_out)
        encoderUp = self.transUp(encoder_out)
        feconv1out = self.feconv1(out)

        combine_out = ndarray.concat(encoderUp, feconv1out, dim=1)
        output = self.decodeConv3(combine_out)
        output = self.Up4(output)

        return output
```











# Note #
training current model on VOC is ongoing 
i will keep going on Xception model and Atrous Conv, and fine tuning on benchmark dataset


