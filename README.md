# TensorFlow_VGG_train_test
Implementation of training and testing process of VGG16 and VGG19 in TensorFlow

## Pre-requisites
* TensorFlow
* Python 2.7.x 
* Pre-trained VGG16 model parameters [`VGG16_modelParams.tensorflow`](https://www.dropbox.com/s/dode6mcjqpqhh4m/VGG16_modelParams.tensorflow?dl=0)

## Easy Run Testing
The testing code `testing.py` test on a image of [weasel](https://github.com/ZZUTK/TensorFlow_VGG_train_test/blob/master/laska.png) using the pre-traind VGG16 model parameters provided by [Davi Frossard](http://www.cs.toronto.edu/~frossard/post/vgg16/). We saved the model parameters to a tensorflow file [`VGG16_modelParams.tensorflow`](https://www.dropbox.com/s/dode6mcjqpqhh4m/VGG16_modelParams.tensorflow?dl=0). 
Download the [model](https://www.dropbox.com/s/dode6mcjqpqhh4m/VGG16_modelParams.tensorflow?dl=0) and run `testing.py`.

    $ python testing.py


## Training
Download the [dataset](https://github.com/jaberg/skdata)
