import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class SigmoidNeuralNetwork:

    def __init__(self, input_layer_size, hidden_layers_size, output_layer_size):
        self.layer_sizes = [input_layer_size] + hidden_layers_size + [output_layer_size]
        self.num_layers = len(self.layer_sizes)
        self.weights = [np.random.rand(layer1, layer2)
                        for layer1,layer2 in zip(self.layer_sizes, self.layer_sizes[1:])]

    def forward(self, input):
        self.layers = [input]
        prev_layer = self.layers[0]
        for weights in self.weights:
            self.layers.append(sigmoid(np.dot(prev_layer,weights)))
            prev_layer = self.layers[-1]



sigmoid_network = SigmoidNeuralNetwork(2,[3],1)
#sigmoid_network.weights = [np.array([[1,2,3],[4,5,6]]),np.array([[7],[8],[9]])]
sigmoid_network.forward(np.array([0.1,0.2]))
print(sigmoid_network.weights)
print(sigmoid_network.layers)
