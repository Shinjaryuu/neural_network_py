import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidprime(x):
    return np.exp(-1) / (1+np.exp(-x))**2

class SigmoidNeuralNetwork:

    def __init__(self, input_layer_size, hidden_layers_size, output_layer_size):
        self.layer_sizes = [input_layer_size] + hidden_layers_size + [output_layer_size]
        self.num_layers = len(self.layer_sizes)
        self.weights = [np.random.rand(layer1, layer2)
                        for layer1,layer2 in zip(self.layer_sizes, self.layer_sizes[1:])]

    def forward(self, input):
        self.ps = [input]
        self.zs = []
        prev_layer = self.ps[0]
        for weights in self.weights:
            self.zs.append(np.dot(prev_layer,weights))
            self.ps.append(sigmoid(self.zs[-1]))
            prev_layer = self.ps[-1]
        return self.ps[-1]

    def cost_function(self, outputs):
        error = 1/2*sum(np.dot(result-output,result-output) for result, output in zip(self.ps[-1], outputs))
        return error


    def cost_function_gradient(self, outputs):
        gradients = []
        delta = [np.mulitply(outputs - self.ps[-1], signmoidprime(self.zs[-1]))]
        for a, z, w in zip(reversed(self.ps[:-1]), reversed(self.zs), reversed(self.weights)):
            delta = np.multiply(np.dot(prev_delta, w.T), sigmoidprime(z))
            gradients.append(np.multiply(a.T, delta))
        return reversed(gradients)



class Test:
    def __init__(self):
        self.sigmoid_network = SigmoidNeuralNetwork(2,[3],1)
        self.sigmoid_network.weights = [np.array([[1.6,2.2,3.7],[1.3,1.9,3.2]]),np.array([[2.5],[4.2],[2.1]])]
        self.inputs = [np.array([0.1,0.2]),np.array([0.3,0.4])]

    def forward(self):
        self.sigmoid_network.forward(self.inputs)
        if abs(self.sigmoid_network.ps[-1][0][0] -0.996858182) < 0.0000001:
            print('Forward passed')
        else:
            print('Forward failed')

    def cost_function(self):
        cost = self.sigmoid_network.cost_function([np.array([0.98]),np.array([0.997])])
        if abs(cost - 0.000144159) < 0.000001:
            print('Cost_function passed')
        else:
            print('Cost_function failed')

    def cost_function_gradient(self):
        epsilon = 1e-4
        network_plus = self.sigmoid_network = SigmoidNeuralNetwork(2,[3],1)
        network_plus.weights = self.sigmoid_network.weights + epsilon
        network_plus.forward(self.inputs)
        network_minus = self.sigmoid_network = SigmoidNeuralNetwork(2,[3],1)
        network_minus.weights = self.sigmoid_network.weights - epsilon
        network_minus.forward(self.inputs)



test = Test()
test.forward()
test.cost_function()
