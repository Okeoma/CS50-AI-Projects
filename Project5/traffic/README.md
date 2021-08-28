# Project 5: Traffic

##This is Project 5: CS50's Introduction to Artificial Intelligence with Python.

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
 
 
