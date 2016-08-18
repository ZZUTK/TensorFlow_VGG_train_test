# TensorFlow_VGG_train_test
Implementation of training and testing process of VGG16 in TensorFlow

* [Pre-requisites](#Prerequisites)
* [Easy Run Testing](#EasyRun)
* [Training on CIFAR-10 dataset](#Training)
* [Files Overview](#Files)

![Demo show](https://github.com/ZZUTK/TensorFlow_VGG_train_test/blob/master/images/Slide2.PNG)


<a name='Prerequisites'>
## Pre-requisites
* TensorFlow
* Python 2.7.x 
* Pre-trained VGG16 model parameters [`VGG16_modelParams.tensorflow`](https://www.dropbox.com/s/9ii6whoj3q3o0cg/VGG16_modelParams.tensorflow?dl=0)
* Training dataset [CIFAR-10 dataset](https://www.dropbox.com/s/3ez7b00be8leqe6/CIFAR-10.dataset.npz?dl=0)

<a name='EasyRun'>
## Easy Run Testing
The testing code [`testing.py`](https://github.com/ZZUTK/TensorFlow_VGG_train_test/blob/master/testing.py) test on a image of [weasel](https://github.com/ZZUTK/TensorFlow_VGG_train_test/blob/master/laska.png) using the pre-traind VGG16 model parameters provided by [Davi Frossard](http://www.cs.toronto.edu/~frossard/post/vgg16/). We saved the model parameters as a tensorflow readable file [`VGG16_modelParams.tensorflow`](https://www.dropbox.com/s/9ii6whoj3q3o0cg/VGG16_modelParams.tensorflow?dl=0). 
Download the [parameter file](https://www.dropbox.com/s/9ii6whoj3q3o0cg/VGG16_modelParams.tensorflow?dl=0) to the same folder and run

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
## Training on CIFAR-10 dataset (in debugging)
The training code [`training.py`](https://github.com/ZZUTK/TensorFlow_VGG_train_test/blob/master/training.py) shows a demo of the training process in TensorFlow. Download the [CIFAR-10 dataset](https://www.dropbox.com/s/3ez7b00be8leqe6/CIFAR-10.dataset.npz?dl=0) to the same folder and run
    
    $ python training.py
    
The print information during traing will be like the follow. Of course, the results may vary with the initalization. The epoch number is set to 10 to get a quick result. For the same reason, we trained on a small model (3 layers) rather than on VGG16 model. Although the accuracy is not good enough, it roughly shows the trend of lower loss and higher accuracy with more epoch. 
```
Loading dataset ...
Color center of images: [ 0.59276742  0.57485735  0.44807526]
Training ...
...
(GPU Info)
...
Epoch  1/10:
	Train Loss = 2.33	 Accuracy = 18.00%
	Valid Loss = 2.13	 Accuracy = 25.00%
Epoch  2/10:
	Train Loss = 2.08	 Accuracy = 20.00%
	Valid Loss = 1.95	 Accuracy = 33.30%
Epoch  3/10:
	Train Loss = 1.99	 Accuracy = 26.00%
	Valid Loss = 1.85	 Accuracy = 34.70%
Epoch  4/10:
	Train Loss = 1.80	 Accuracy = 40.00%
	Valid Loss = 1.79	 Accuracy = 37.20%
Epoch  5/10:
	Train Loss = 1.47	 Accuracy = 44.00%
	Valid Loss = 1.66	 Accuracy = 41.10%
Epoch  6/10:
	Train Loss = 1.25	 Accuracy = 54.00%
	Valid Loss = 1.61	 Accuracy = 43.60%
Epoch  7/10:
	Train Loss = 1.24	 Accuracy = 60.00%
	Valid Loss = 1.68	 Accuracy = 43.20%
Epoch  8/10:
	Train Loss = 1.06	 Accuracy = 64.00%
	Valid Loss = 1.62	 Accuracy = 46.00%
Epoch  9/10:
	Train Loss = 0.98	 Accuracy = 72.00%
	Valid Loss = 1.55	 Accuracy = 47.00%
Epoch 10/10:
	Train Loss = 0.90	 Accuracy = 76.00%
	Valid Loss = 1.60	 Accuracy = 45.40%
```


    
<a name='Files'>
## Files Overview
* [`layerConstructor.py`](https://github.com/ZZUTK/TensorFlow_VGG_train_test/blob/master/layerConstructor.py) provides higher level functions to build basic convolution, pooling, and fully connection layers.
* [`VGG16_model.py`](https://github.com/ZZUTK/TensorFlow_VGG_train_test/blob/master/VGG16_model.py) build the VGG16 model, respectively, using [`layerConstructor.py`](https://github.com/ZZUTK/TensorFlow_VGG_train_test/blob/master/layerConstructor.py)
* [`imagenet_classes.py`](https://github.com/ZZUTK/TensorFlow_VGG_train_test/blob/master/imagenet_classes.py) stores the class names of the ImageNet dataset, cited from [Davi Frossard](http://www.cs.toronto.edu/~frossard/post/vgg16/)
* [`weasel.png`](https://github.com/ZZUTK/TensorFlow_VGG_train_test/blob/master/weasel.png) is an example image for testing
* [`training.py`](https://github.com/ZZUTK/TensorFlow_VGG_train_test/blob/master/training.py) and [`testing.py`](https://github.com/ZZUTK/TensorFlow_VGG_train_test/blob/master/testing.py) are demos of traing and testing

