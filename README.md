# TensorFlow_VGG_train_test
Implementation of training and testing process of VGG16 and VGG19 in TensorFlow

* [Pre-requisites](#Pre_requisites)
* [Easy Run Testing](#EasyRun)
* [Training on CIFAR-10 dataset](#Training)
* [Files Overview](#Files)

<a name='Pre-requisites'>
## Pre-requisites
* TensorFlow
* Python 2.7.x 
* Pre-trained VGG16 model parameters [`VGG16_modelParams.tensorflow`](https://www.dropbox.com/s/9ii6whoj3q3o0cg/VGG16_modelParams.tensorflow?dl=0)
* Training dataset [CIFAR-10 dataset](https://www.dropbox.com/s/3ez7b00be8leqe6/CIFAR-10.dataset.npz?dl=0)

<a name='EasyRun'>
## Easy Run Testing
The testing code `testing.py` test on a image of [weasel](https://github.com/ZZUTK/TensorFlow_VGG_train_test/blob/master/laska.png) using the pre-traind VGG16 model parameters provided by [Davi Frossard](http://www.cs.toronto.edu/~frossard/post/vgg16/). We saved the model parameters as a tensorflow readable file [`VGG16_modelParams.tensorflow`](https://www.dropbox.com/s/9ii6whoj3q3o0cg/VGG16_modelParams.tensorflow?dl=0). 
Download the [parameter file](https://www.dropbox.com/s/9ii6whoj3q3o0cg/VGG16_modelParams.tensorflow?dl=0) to the same folder and run `testing.py`.

    $ python testing.py

Testing result
```
Restoring VGG16 model parameters ...

Classification Result:
Category Name: weasel 
Probability: 88.05%

Category Name: polecat, fitch, foulmart, foumart, Mustela putorius 
Probability: 6.82%

Category Name: mink 
Probability: 4.79%

```

<a name='Training'>
## Training on CIFAR-10 dataset
The training code `training.py` shows a demo of the training process in TensorFlow. Download the [CIFAR-10 dataset](https://www.dropbox.com/s/3ez7b00be8leqe6/CIFAR-10.dataset.npz?dl=0) to the same folder and run `training.py`
    
    $ python training.py
    
    
<a name='Files'>
## Files Overview
* [`layerConstructor.py`](https://github.com/ZZUTK/TensorFlow_VGG_train_test/blob/master/layerConstructor.py) provides higher level functions to build basic convolution, pooling, and fully connection layers.
* [`VGG16_model.py`](https://github.com/ZZUTK/TensorFlow_VGG_train_test/blob/master/VGG16_model.py) and [`VGG19_model.py`](https://github.com/ZZUTK/TensorFlow_VGG_train_test/blob/master/VGG19_model.py) build the VGG16 and VGG19 model, respectively, using `layerConstructor.py`
* `imagenet_classes.py` stores the class names of the ImageNet dataset. This file is stolen from [Davi Frossard](http://www.cs.toronto.edu/~frossard/post/vgg16/)

