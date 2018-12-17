# cs480 HW4
# Modified By Nick Flouty
# Original: https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
# Output:
#   Input weights: The value of how likely this neuron with this weight is to an output. [theres two outputs, 1 or 0]
#   Output weights:  The value of the weights between the hidden layer to the output. [From 1 and 0 to y] 
'''
Problem:
Consider the Boolean function F(a0, a1, a2, b0, b1, b2) on 6 variables that takes value 1 if
the integer represented by the first three bits a0a1a2 is larger than the integer represented
by the last three bits b0b1b2. (For example, F(1, 0, 0, 0, 1, 1) is 1 since 100 represents
integer 4, and 011 represents integer 3. Train a neural network with a single hidden
layer with two neurons, and an output layer with one neuron. Choose a random initial
weights for neurons, and after each epoch, test on all the 64 inputs and report the best
success ratio found after 1000 epochs are completed.
'''
import numpy as np
import itertools

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = 2*np.random.rand(self.input.shape[1],2) - 1 # array [sizeofinput(6 in our case), # of neurons in hidden layer(2 in our case, either 1 or 0)]
        self.weights2   = 2*np.random.rand(2,1) - 1         # The 2* () - 1 makes the random value have a mean of 0. which made a huge impact on our output
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2



if __name__ == "__main__":
    x = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 1],
                    [0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 1], [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1], [0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 1],
                    [0, 1, 0, 1, 0, 0], [0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 1, 0], [0, 1, 0, 1, 1, 1], [0, 1, 1, 0, 0, 0],
                    [0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 1, 0], [0, 1, 1, 0, 1, 1], [0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 0, 1], 
                    [0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 1, 1], [1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 1, 1],
                    [1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0],
                    [1, 0, 1, 1, 0, 1], [1, 0, 1, 1, 1, 0], [1, 0, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1], 
                    [1, 1, 0, 0, 1, 0], [1, 1, 0, 0, 1, 1], [1, 1, 0, 1, 0, 0], [1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 1, 0],
                    [1, 1, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 1], [1, 1, 1, 0, 1, 0], [1, 1, 1, 0, 1, 1],
                    [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
    y = np.array([[0],[0],[0],[0],[0],
				  [0],[0],[0],[1],[0],
				  [0],[0],[0],[0],[0],
				  [0],[1],[1],[0],[0],
				  [0],[0],[0],[0],[1],
				  [1],[1],[0],[0],[0],
				  [0],[0],[1],[1],[1],
				  [1],[0],[0],[0],[0],
				  [1],[1],[1],[1],[1],
				  [0],[0],[0],[1],[1],
				  [1],[1],[1],[1],[0],
				  [0],[1],[1],[1],[1],
				  [1],[1],[1],[0]
				  ])
    nn = NeuralNetwork(x,y)
    
    best = 0
    final = []
    weights= []
    weights2 = []
    for i in range(1000):   # for each epoch
        nn.feedforward()
        nn.backprop()

        success = 0
        for j in range(len(nn.output)):
            if(nn.output[j] >= .5):
                prediction = 1
            else:
                prediction = 0

            if prediction == nn.y[j]:
                success += 1

        if success > best:
            best = success
            index = i
            weights = nn.weights1
            weights2 = nn.weights2
            final = nn.output

           
    ratio = (best / len(nn.output)) * 100
    print("Optimal Success Ratio: ",ratio,'%')
    print("Epoch: ",index)
    print("Input Weights : \n", weights)
    print("Output Weight : \n", weights2)
