# Project 5: Traffic

Write an AI to identify which traffic sign appears in a photograph.

```
$ python traffic.py gtsrb
Train on 15984 samples
Epoch 1/10
15984/15984 [==============================] - 10s 623us/sample - loss: 2.8565 - accuracy: 0.3022
Epoch 2/10
15984/15984 [==============================] - 8s 510us/sample - loss: 1.3484 - accuracy: 0.5951
Epoch 3/10
15984/15984 [==============================] - 8s 531us/sample - loss: 0.8283 - accuracy: 0.7494
Epoch 4/10
15984/15984 [==============================] - 12s 736us/sample - loss: 0.5758 - accuracy: 0.8270
Epoch 5/10
15984/15984 [==============================] - 12s 744us/sample - loss: 0.4241 - accuracy: 0.8725
Epoch 6/10
15984/15984 [==============================] - 10s 602us/sample - loss: 0.3391 - accuracy: 0.8956
Epoch 7/10
15984/15984 [==============================] - 10s 620us/sample - loss: 0.3102 - accuracy: 0.9103
Epoch 8/10
15984/15984 [==============================] - 11s 668us/sample - loss: 0.2747 - accuracy: 0.9207
Epoch 9/10
15984/15984 [==============================] - 10s 614us/sample - loss: 0.2208 - accuracy: 0.9362
Epoch 10/10
15984/15984 [==============================] - 8s 528us/sample - loss: 0.1961 - accuracy: 0.9418
10656/10656 - 2s - loss: 0.1392 - accuracy: 0.9606
```

## Background

As research continues in the development of self-driving cars, one of the key challenges is computer vision, allowing these cars to develop an understanding of their environment from digital images. In particular, this involves the ability to recognize and distinguish road signs – stop signs, speed limit signs, yield signs, and more.

In this project, you’ll use TensorFlow to build a neural network to classify road signs based on an image of those signs. To do so, you’ll need a labeled dataset: a collection of images that have already been categorized by the road sign represented in them.

Several such data sets exist, but for this project, we’ll use the [German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) (GTSRB) dataset, which contains thousands of images of 43 different kinds of road signs.

## Specification

Complete the implementation of `load_data` and `get_model` in `traffic.py`.

- The `load_data` function should accept as an argument `data_dir`, representing the path to a directory where the data is stored, and return image arrays and labels for each image in the data set.
    - You may assume that `data_dir` will contain one directory named after each category, numbered `0` through `NUM_CATEGORIES - 1`. Inside each category directory will be some number of image files.
    - Use the OpenCV-Python module (`cv2`) to read each image as a `numpy.ndarray` (a `numpy` multidimensional array). To pass these images into a neural network, the images will need to be the same size, so be sure to resize each image to have width `IMG_WIDTH` and height `IMG_HEIGHT`.
    - The function should return a tuple `(images, labels)`. `images` should be a list of all of the images in the data set, where each image is represented as a `numpy.ndarray` of the appropriate size. `labels` should be a list of integers, representing the category number for each of the corresponding images in the `images` list.
    - Your function should be platform-independent: that is to say, it should work regardless of operating system. Note that on macOS, the `/` character is used to separate path components, while the `\` character is used on Windows. Use `os.sep` and `os.path.join` as needed instead of using your platform’s specific separator character.
- The `get_model` function should return a compiled neural network model.
    - You may assume that the input to the neural network will be of the shape `(IMG_WIDTH, IMG_HEIGHT, 3)` (that is, an array representing an image of width `IMG_WIDTH`, height `IMG_HEIGHT`, and `3` values for each pixel for red, green, and blue).
    - The output layer of the neural network should have `NUM_CATEGORIES` units, one for each of the traffic sign categories.
    - The number of layers and the types of layers you include in between are up to you. You may wish to experiment with:
        - different numbers of convolutional and pooling layers
        - different numbers and sizes of filters for convolutional layers
        - different pool sizes for pooling layers
        - different numbers and sizes of hidden layers
        - dropout

Ultimately, much of this project is about exploring documentation and investigating different options in `cv2` and `tensorflow` and seeing what results you get when you try them!

You should not modify anything else in `traffic.py` other than the functions the specification calls for you to implement, though you may write additional functions and/or import other Python standard library modules. You may also import `numpy` or `pandas`, if familiar with them, but you should not use any other third-party Python modules. You may modify the global variables defined at the top of the file to test your program with other values.

## Acknowledgements

Data provided by [J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. The German Traffic Sign Recognition Benchmark: A multi-class classification competition. In Proceedings of the IEEE International Joint Conference on Neural Networks, pages 1453–1460. 2011](http://benchmark.ini.rub.de/index.php?section=gtsrb&subsection=dataset#Acknowledgements)

# Project 5: Traffic Solution

##This is Solutions to Project 5: CS50's Introduction to Artificial Intelligence with Python.

The goal of project is to write an AI that identifies a sign that shows in a photograph.

- A path to a directory of photos are loaded and the AI is expected to return arrays of images and labels in the data set.
- When the images are read from the array, they are passed into a neural network with all image having same size and the 
AI will then return the corresponding images and labels.
- the output layer of the neural network must have the correct number of categories or units of each traffic sign.
- In generating the output results, I experimented with a number of layered components such as:
 different numbers of convolutional and pooling layers 
 different numbers and sizes of filters for convolutional layers 
 different pool sizes for pooling layers 
 different numbers and sizes of hidden layers
 dropout.

- My findings are:
* Adding a Convolutional layer of 32 filters using a 3x3 kernel was ideal for the neural network.
* A max-pooling layer with 3x3 pool size was also essentially ideal for reducing the pixel size of the images to a 
more efficient input feed for the neural network.
* After flattening the units into the neural network, I tried using different numbers of hidden layers, I noticed 
that reducing the numbers of layers to only one or two and also reducing the size of units can greatly affect the
 accuracy and loss values drastically. Lowering the number and size of hidden layers reduces the accuracy 
 and increases the loss value which is not what we want.
 Likewise, increasing the number and size of hidden layers so much gives a reduced accuracy and not too favourable loss value.
 The ideal number used for the project as hidden layers is three, while their sizes ranges from multiplying the number
 of categories of units by 40, 20 and 10 respectively. The result of this gave accuracy close to 100% and
  loss value almost zero (0).
 
 My Project's YouTube Video link: https://youtu.be/lL-VwE3J4pw
 
