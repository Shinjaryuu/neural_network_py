import numpy as np

class SigmoidNeuralNetwork:

    def __init__(self, input_layer_size, hidden_layers_size, output_layer_size):
        self.layers = [np.array([0.0 for i in range(input_layer_size)]),
                            np.array([0.0 for i in range(hidden_layers_size[layer])])
                            for layer in len(hidden_layers_size)],
                            np.array([0.0 for i in range(output_layer_size)])
        self.weights = [np.random.random(len(layer1),len(layer2))
                        for layer1,layer2 in self.layers.zip(self.layers[1:])]

    @staticmethod
    def signoid(x):
        return 
