from tensorflow.keras.datasets import mnist
from numpy import *
from numpy import exp as exxp


def clamp(x):
    if x > 1: x = 1
    elif x < 0: x = 0
    return x

class Layer(): 
    def __init__(self, current_size, next_size) -> None:
        self.next_size = next_size
        self.current_size = current_size
        self.weights = random.randn(self.next_size, self.current_size)
        self.biases = random.randn(self.next_size)
    def __repr__(self) -> str:
        return str(self.current_size)
        
    def feed_forward(self, inputs):
        out = []
        for i in range(self.next_size):
            weighted_sum = self.biases[i]
            for idx, inp in enumerate(inputs):
                weighted_sum += inp * self.weights[i][idx]
                
            out.append(self.activation(weighted_sum) )                
                
            
        return out
    def activation(self, x):
        x = float128(x)
        return 1/ (1 + exxp(-x))

class NeuralNetwork():
    def __init__(self, layer_config) -> None:
        self.layers = []
        layer_config.append(layer_config[-1])
        for idx in range(len(layer_config) - 1):
            self.layers.append(Layer(layer_config[idx], layer_config[idx+1]))
    def calculate(self, inputs):
        for layer in self.layers:
            inputs = layer.feed_forward(inputs)
        return inputs
    def classify(self, inputs):
        predictions = self.calculate(inputs)
        highest_val = -1000
        heighest_idx = 0
        for i, value in enumerate(predictions):
            if value > highest_val:
                highest_val = value
                heighest_idx = i
        return heighest_idx, highest_val
                
            
(train_X, train_y), (test_X, test_y) = mnist.load_data()

neural_network = NeuralNetwork([784,16,16,10])

mega_train=train_X[0][0]
for i in range(len(train_X[0])-1):
    
    mega_train = mega_train+train_X[0][i+1]

print(neural_network.classify(mega_train))