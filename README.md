# Readme
## Real Time Doodle Classifier
In this project, a Real Time Doodle Classifier is developed using elementary concepts of Deep Learning and Convolutional Neural Networks.
![](https://i.imgur.com/lxaAiNA.gif)
## Dataset
The Neural Network Model was trained on a dataset of 2045200 images and tested on dataset of 761837 containing images of 20 differnt classes.
## Approach
***The Prelimnary Stage*** involved studying and learning the basics of Machine Learning and Deep Learning algorithms.

***Coding Networks from scratch of a digit classifier*** using numpy library and writing all the functions for forward and backward passes along with different activation functions .And ran the model for different values of learning rate, hyperparameters,batchsize and no. of epochs for better understanding of the subject.

***Coding the cnn model of the network*** by using torch for convolution of image with filters along with maxpooling and padding. After multiple convolutional layers, the input representation is flattened into a feature vector and passed through a network of neurons to predict the output probabilities.




***Drawing Pad*** is a model developed to facilitate the user in drawing doodle onto a Drawing Pad of 500x500pi by using the mouse events available in OpenCV and then resizing the image to 28x28pi and passing it to the trained CNN model for classification
## Libraries required
1. Pytorch()

2. OpenCV (For creating the drawing pad)

3. Matplotlib (to plot graphs and display images)

4. Numpy

## CNN MODEL
### Architecture

***convolution layer***




| Layers | Kernal Size | Filters | Maxpool | Padding |
| ------ | ----------- | ------- | ------- | ------- |
| conv1       |  (5,5)           |   6      |      (2,2)   |   1      |
|   conv2     |     (5,5)        | 16   | (2,2)    |  1  |


***Fully Connected Layer***


| Layer | Size |
| -------- | -------- |
|    FC1      |  16x4x4,120        |
|    FC2      |    120,84      |
| FC3     | T84,20     |


### Hyper parameters



| Parameters    | Values |
| ------------- | ------ |
| Learning rate | 0.015  |
| Epochs        | 75     |
| Batch size    | 5113   |
| Beta          | 0.9    |
| Optimizer     | SGD    |

## Output

![](https://i.imgur.com/jw0jL5B.jpg)



| Dataset | Accuracy |
| -------- | -------- |
|      Train    |     88.98     |
| Test     | 88.65     |






