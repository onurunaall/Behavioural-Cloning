# Behavioural-Cloning

For an autonomous car, it is significant to avoid the objects. With behavioural technique, it is possible to clone a human behaviour for driving. The first part which is classifying the data is, in fact, a look to data science. In this part, the train and test data are split from each other and then they are classified. Also the density of direction of cloned behaviour of driver is listed and classfied. In second part the data which are the images where car captured during recording the behaviour. Then, it is possible to use these images for locating where exactly the car on the road. After that using some visual augmentation techniques, the data is preprocessed to be use properly by neural network. At the last part, there is a Convolutional Neural Network structure which is an example of Nvidia's model.

The whole code can be found as behavioral_cloning.py

For the action server which is the script that connects the behavioural cloning code and the Udacity similator, please look at drive.py.
