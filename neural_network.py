import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidprime(x):
    return np.exp(-x) / (1+np.exp(-x))**2

class SigmoidNeuralNetwork:
    """This is a neural network with a fixed number of neurons and layers.
    The number of neurons and layers is determined at intialization.
    The activation function for each neuron is a sigmoidself.
    The error is the sum of squares"""

    def __init__(self, input_layer_size, hidden_layers_size, output_layer_size):
        """Create a neural network with the given number of neurons per layer
        input_layer_size: number of input neurons
        hidden_layers_size: list of number of neurons for each hidden layer
        output_layer_size: number of output neurons"""
        self.layer_sizes = [input_layer_size] + hidden_layers_size + [output_layer_size]
        self.num_layers = len(self.layer_sizes)
        self.weights = [np.random.rand(layer1, layer2)
                        for layer1,layer2 in zip(self.layer_sizes, self.layer_sizes[1:])]

    def forward(self, input):
        """Feed the input to the network, updating all the neurons"""
        #ps: value of the neuron post activation function
        self.ps = [input]
        #zs: values of the neurons pre activation function
        self.zs = []
        prev_layer = self.ps[0]
        for weights in self.weights:
            self.zs.append(np.dot(prev_layer,weights))
            self.ps.append(sigmoid(self.zs[-1]))
            prev_layer = self.ps[-1]
        return self.ps[-1]

    def cost_function(self, outputs):
        """Compute the cost for a given output. Use after using forward.
        This neural network uses the sum of squares error"""
        error = 1/2*sum(np.dot(result-output,result-output) for result, output in zip(self.ps[-1], outputs))
        return error

    def get_params(self):
        """Returns a flattened numpy array containing the weights.
        The first weights correspond to the weights between layer and layer 2,
        the next between layer 2 and layer 3 etc"""
        return np.concatenate([weights.ravel() for weights in self.weights])

    def set_params(self, params):
        """Sets the weights of the neural network equal to the params
        params is a flat numpy array."""
        start = 0
        for index, [size1, size2] in enumerate(zip(self.layer_sizes, self.layer_sizes[1:])):
            end = start + size1 * size2
            self.weights[index] = np.reshape(params[start:end], (size1, size2))
            start = end

    def cost_function_gradient(self, outputs):
        """Compute the gradient of the cost function for given outputs"""
        gradients = []
        delta = np.multiply(-np.array(outputs) + self.ps[-1], sigmoidprime(self.zs[-1]))
        for p, z, w in zip(reversed(self.ps[:-1]), reversed(self.zs[:-1]), reversed(self.weights)):

            gradients.append(np.dot(np.transpose(p), delta))
            delta = np.multiply(np.dot(delta, w.T), sigmoidprime(z))
        gradients.append(np.dot(np.transpose(self.ps[0]), delta))
        return reversed(gradients)

    def compute_gradients(self, outputs):
        gradients = self.cost_function_gradient(outputs)
        return np.concatenate([gradient.ravel() for gradient in gradients])



class Test:
    def __init__(self):
        self.sigmoid_network = SigmoidNeuralNetwork(2,[3],1)
        self.sigmoid_network.weights = [np.array([[1.6,2.2,3.7],[1.3,1.9,3.2]]),np.array([[2.5],[4.2],[2.1]])]
        self.inputs = [np.array([0.1,0.2]),np.array([0.3,0.4])]
        self.outputs = [[0.3], [0.6]]

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

    def params(self):
        params_init = self.sigmoid_network.weights
        self.sigmoid_network.set_params(self.sigmoid_network.get_params())
        if params_init == self.sigmoid_network.weights:
            print('Params passed')
        else:
            print('Params failed')

    def cost_function_gradient(self):
        epsilon = 1e-4
        params = self.sigmoid_network.get_params()
        num_grads = np.zeros(len(params))
        perturbation = np.zeros(len(params))

        for p in range(len(params)):
            perturbation[p] = epsilon
            self.sigmoid_network.set_params(params + perturbation)
            self.sigmoid_network.forward(self.inputs)
            cost_plus = self.sigmoid_network.cost_function(self.outputs)
            self.sigmoid_network.set_params(params - perturbation)
            self.sigmoid_network.forward(self.inputs)
            cost_minus = self.sigmoid_network.cost_function(self.outputs)
            num_grads[p] = (cost_plus-cost_minus)/(2*epsilon)
            perturbation[p] = 0

        self.sigmoid_network.set_params(params)
        self.sigmoid_network.forward(self.inputs)
        grads = self.sigmoid_network.compute_gradients(self.outputs)
        
        if sum(abs(num_grad-grad) for num_grad,grad in zip(num_grads, grads)) < 0.0001:
            print('Cost_function_gradient passed')
        else:
            print('Cost_function_gradient failed')



test = Test()
test.forward()
test.cost_function()
test.params()
test.cost_function_gradient()
