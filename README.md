# Neural Network to Recognize MNIST Digits

Implement one of the common machine learning algorithms: Neural Networks. I will train and test a neural network with the dataset as provided and experiment with different settings of hyper parameters. The neural network learning algorithm is implemented from scratch - not a single machine learning library is used. Only the Python standard library and numpy is used.


Dataset
The dataset is the MNIST database. It is a database of handwritten digits. The dataset is split into training set and test set stored in five csv.gz files.

There are 50,000 training samples in TrainDigitX.csv.gz, 10,000 test samples in TestDigitX.csv.gz, and another 5,000 test sample in TestDigitX2.csv.gz. Each sample is a handwritten digit represented by a 28 by 28 greyscale pixel image. Each pixel is a value between 0 and 1 with a value of 0 indicating white. Each sample used in the dataset (a row in TrainDigitX.csv, TestDigitX.csv, or TestDigitX2.csv.gz) is a feature vector of length 784(28x28=784). TrainDIgitY.csv.gz and TestDigitY.csv.gz provide labels for samples in TrainDigitX.csv.gz and TestDigitX.csv.gz, respectively. The value of a label is the digit it represents, e.g, a label of value 8 indicates the sample represents the digit 8.

Note: The data files are compressed. The loadtxt() method from numpy can read compressed or uncompressed csv files.

The python script first creates a neural network of specified size and runs stochastic gradient descent on a cost function over given training data. We assume a network with 3 layers: one input layer, one hidden layer and one output layer. The main file is named as neural_network.py which accepts seven arguments. It can be run on the command line in the following manner:

`python neural network.py NInput NHidden NOutput TrainDigitX.csv.gz TrainDig- itY.csv.gz TestDigitX.csv.gz PredictDigitY.csv.gz`

The code then trains a neural net(NInput: number of neurons in the input layer, NHidden: number of neurons in the hidder layer, NOutpuy: number of neurons in out- put layer) using the training set TrainDigitX.csv.gz and TrainDigitY.csv.gz, and then makes predictions for all the samples in TestDigitX.csv.gz and output the labels to PredictDigitY.csv.gz.

I have set the default value of number of epochs to 30, size of mini-batches to 20, and learning rate to 3 respectively.

The nonlinearity used in my neural net is the basic sigmoid function. σw,b(x) = 1 / 1 + e−(w.x+b)

I have used mini-batches to train the neural net for several epochs. Mini-batches are just the training dataset divided randomly into smaller sets to approximate the gradient. The main steps of training a neural net using stochastic gradient descent are:
- Assign random initial weights and biases to the neurons. Each initial weight or bias is a random floating-point number drawn from the standard normal distribution (mean 0 and variance 1).
- For each training example in a mini-batch, use backpropagation to calculate a gradient estimate, which consists of following steps:
  1. Feed forward the input to get the activations of the output layer.
  2. Calculate derivatives of the cost function for that input with respect to the activations of the output layer.
  3. Calculate the errors for all the weights and biases of the neurons using backpropogation.
- Update weights (and biases) using stochastic gradient descent
- Repeat this for all mini-batches. Repeat the whole process for specified number of epochs. At the end of each epoch I evaluate the network on the test data and display its accuracy.
- I use the quadratic cost function.

Example usage:
1. Create a neural net of size [784,30,10]. This network has three layers: 784 neurons in the input layer, 30 neurons in the hidden layer, and 10 neurons in the output layer. Then train it on the training data for 30 epochs, with a minibatch size of 20 and η = 3.0 and test it on the test data(TestDigitX.csv.gz).
2. Try different hyperparameter settings(number of epochs, η, and mini-batch size, etc.).
3. Replace the quadratic cost function by a cross entropy cost function.
4. L2 regularization. Use L2 regularizers on the weights to modify the cost function.
