###################
# A multilayer perceptron (neural network) implementation
# Good to use for understanding how neural nets works.
# Author: John Berroa
###################

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pickle
from datetime import datetime as dt
# from testdata import ToyData
from scipy.special import expit as sigmoid

# what I learned: build it simple first then add feautures; don't try to debug on a complex system to get the simplest model to work
#TODO: Allow for custom activation function
#TODO: Error functions?
#TODO: Momentum?
#TODO: "Take the result with the best training or validation performance" so have an option to train it multiple times

class MultiLayerPerceptron:
    """
    Creates an MLP with customizable parameters.
    Stores weight types, epsilons (learning rates), and activation functions for each layer so that they can be
    different for experimentation
    """

    ################ Magic Functions ################

    def __init__(self, global_activation, global_epsilon, global_weight_type, debug=False):
        if debug:
            np.random.seed(777)
        self.global_weight_type = self._init_weights(global_weight_type)
        self.global_epsilon = global_epsilon
        self.global_activation_func = self._init_activation(global_activation)

        # Description of the network
        self.layer_weights = []
        self.layer_weight_types = []
        self.layer_activation_funcs = []  # this is kept as a string for ease of reading
        self.layer_epsilons = []
        self.layer_sizes = []

        # Variables needed for training
        self.layer_logits = []
        self.layer_outputs = []

        # Recorded variables for analysis
        self.errors = []
        self.epsilon_over_time = []


    def __str__(self):
        """
        Gives a summary of the structure of the network
        """
        if len(self.layer_sizes) == 0:
            string = "Multilayer Perceptron not yet built, therefore unable to print structure.  Use 'create_layer'."
        else:
            string = "Details of the Multilayer Perceptron:\n" \
                     " Layers: {}\n" \
                     " Structural overview: {} (# neurons/layer)\n" \
                     " Input dimensionality: {}\n" \
                     " Global defaults:\n" \
                     "    Weight type: {}\n" \
                     "    Epsilon: {}\n" \
                     "    Activation function: {}\n" \
                     " Layer settings:\n".format(len(self.layer_sizes), self.layer_sizes, self.layer_weights[0].shape[0]-1,
                                                self.global_weight_type, self.global_epsilon, self.global_activation_func)
            for layer in range(len(self.layer_sizes)):
                layerstring = "   Layer: {}\n" \
                              "    Number of neurons: {}\n" \
                              "    Weight type: {}\n" \
                              "    Epsilon: {}\n" \
                              "    Activation function: {}\n".format(layer+1, self.layer_sizes[layer],
                                                                     self.layer_weight_types[layer], self.layer_epsilons[layer],
                                                                     self.layer_activation_funcs[layer])
                string = string + layerstring

            laststring = "Note: if the class is printed before the network is fully created, it will print " \
                         "whatever has been built thus far."
            string = string + laststring
        return string

    ##############---Magic Functions---##############
    ################ Network Creation Functions ################

    def _init_weights(self, w):
        """
        Checks if input global weight type is valid; creating weights is the _create_weights function
        :param w:
        :return desired weight initialization:
        """
        possible_weights = ['normal', 'trunc', 'ones', 'zeros', 'uniform']
        if w not in possible_weights:
            raise ValueError("Invalid global weight type.  "
                             "Input: '{}'; Required: 'normal','trunc','ones',zeros', or 'uniform'.".format(w))
        else:
            return w


    def _create_weights(self, w, dim_in, size):
        """
        Creates a weight matrix based on the input dimensionality and the number of neurons.  Adds bias dimension
        :param w:
        :param dim_in:
        :param size:
        :return weight matrix:
        """
        if w == 'default':  # if default, reenter function with default weight name type
            return self._create_weights(self.global_weight_type, dim_in, size)
        # +1 because of adding a bias
        elif w == 'normal':
            return np.random.normal(size=(dim_in + 1, size))
        elif w == 'trunc':
            return stats.truncnorm.rvs(-1,1,size=(dim_in + 1, size))
        elif w == 'ones':
            return np.ones((dim_in + 1, size))
        elif w == 'zeros':
            return np.zeros((dim_in + 1, size))
        elif w == 'uniform':
            return np.random.uniform(-1,1,(dim_in + 1, size))
        else:
            raise ValueError("Invalid weight initialization type.  "
                             "Input: '{}'; Required: 'normal','trunc','ones',zeros', or 'uniform'.".format(w))


    def _init_activation(self, a):
        """
        Checks if input activation type is valid
        :param a:
        :return desired function:
        """
        possible_activations = ['sigmoid', 'tanh', 'linear', 'relu']
        if a not in possible_activations:
            raise ValueError("Invalid global activation function.  "
                             "Input: '{}'; Required: 'sigmoid','tanh','linear', or 'relu'.".format(a))
        else:
            return a


    def _get_activation_func(self, request):
        """
        Retrieves a specific activation function if desired, or returns the global one if not defined
        :param request:
        :return activation function:
        """
        if request == 'default':
            return self.global_activation_func
        elif request == 'sigmoid':
            return sigmoid
        elif request == 'tanh':
            return np.tanh
        elif request == 'linear':
            return lambda x: x
        elif request == 'relu':
            return np.vectorize(lambda x: x if x > 0 else 0)
        else:
            raise ValueError("Invalid activation function.  "
                             "Input: '{}'; Required: 'sigmoid','tanh','linear', or 'relu'.".format(request))


    def _get_derivative(self, func):
        """
        Returns the derivative of the sent in activation function, or the derivative of the error function
        :param func:
        :return func's derivative:
        """
        if func == 'sigmoid':
            return lambda out: out * (1 - out)
        elif func == 'tanh':
            return lambda out: 1 - out**2
        elif func == 'linear':
            return 1
        elif func == 'relu':
            return np.vectorize(lambda out: 1 if out > 0 else 0)
        elif func == 'error':
            return lambda output, target: -(output - target) #is the negtive necessary
        else:
            raise ValueError("Invalid activation function to generate derivative.  "
                             "Input: '{}'; Required: 'sigmoid','tanh','linear', 'relu', or 'error'.".format(func))


    def _get_epsilon(self, request):
        """
        Returns a different epsilon if desired
        :param request:
        :return epsilon:
        """
        if request == 'default':
            return self.global_epsilon
        else:
            return request  # because it will be a number in this case


    def create_layer(self, size, dim_in=None, activation='default', weight_type='default', epsilon='default'):
        """
        Create a layer in the network.  Each layer has a size, dimensionality, activation, weights, weight_types, and
        epsilon.  dim_in is used for the first layer only; after that it is implied from previous layers.  Layers are
        purely defined by these parameters in the same position in their own lists i.e. layer 2 = size[2],activation[2],
        weights[2],epsilon[2],...etc.
        :param size:
        :param dim_in:
        :param activation:
        :param weight_type:
        :param epsilon:
        :return:
        """
        epsilon = self._get_epsilon(epsilon)
        if len(self.layer_weights) == 0:  # if first layer, make sure there is input dimensionality provided
            if dim_in == None:
                raise ValueError("You are creating the first layer of the network.  "
                                 "Please provide the input dimensionality with 'dim_in'")
            else:
                weights = self._create_weights(weight_type, dim_in, size)
        else:
            input_dimensionality = self.layer_sizes[-1]  # returns the number of outputs of the previous layer
            weights = self._create_weights(weight_type, input_dimensionality, size)

        # update layer information storage
        self.layer_weights.append(weights)
        self.layer_epsilons.append(epsilon)
        self.layer_sizes.append(size)
        if activation == 'default':
            self.layer_activation_funcs.append(self.global_activation_func)
        else:
            self.layer_activation_funcs.append(activation)
        if weight_type == 'default':
            self.layer_weight_types.append(self.global_weight_type)
        else:
            self.layer_weight_types.append(weight_type)

    ##############---Network Creation Functions---##############
    ################ Training Functions ################

    def _add_bias(self, v):
        """
        Takes in an input vector, adds a bias of 1 to the front of it, and then expands dimensions to avoid:
        (3,) vs. (1,3)
        :param v:
        :return vector with bias and proper dimensionality:
        """
        v = np.append(1, v)
        try:  # add dimension if it doesn't exist
            _ = v.shape[1]
        except:
            v = np.expand_dims(v, axis=0)
        return v


    def _epsilon_decay(self):
        raise NotImplementedError


    def _forward_step(self, layer, input):
        """
        Calculates the given layer's output given an input
        :param layer:
        :param input:
        :return layer output:
        """
        if layer == 0:
            print("SAVING ORIGINAL INPUT+BIAS")
            self.input = self._add_bias(input)
        print("THE INPUT:\n", self._add_bias(input))
        print("THE WEIGHTS:\n", self.layer_weights[layer])
        sums = np.dot(self.input, self.layer_weights[layer])
        print("THE SUMS:\n", sums)
        print("THE SUMS DIMS:\n",sums.shape)
        activation_function = self._get_activation_func(self.layer_activation_funcs[layer])
        # output = []
        output = activation_function(sums)
        # if layer == len(self.layer_sizes)-1:
        #     print("ADDING BIAS TO FINAL LAYER")
        #     output = self._add_bias(output)
        #     sums = self._add_bias(sums)
        #     self.layer_logits.append(sums)
        #     self.layer_outputs.append(output)
        #     print(len(self.layer_outputs))
        # else:
        self.layer_logits.append(sums)  # used for the backprop step   MAY NOT BE USED????
        self.layer_outputs.append(output)
        return output


    def _feedforward(self, input):
        """
        Propagates an input through the network to get the network's result, to be fed into backprop.
        Exact copy of _predict, but in the proper function section and with an easier to understand name in its
        context, as well as no return.
        :param input:
        """
        for layer in range(len(self.layer_sizes)):
            input = self._forward_step(layer, input)


    def _calculate_error(self, output, target):
        """
        Calculates the squared error between the network output and the target, and returns it
        :param output:
        :param target:
        :return error:
        """
        # so this should be a sum, where it sums over one item in the stochastic case, because it'll be a [1], but in the batch case
        # it should be a list of numbers (inputs that were sent through) that it iterates through
        # error = np.mean([.5 * (target - sample) ** 2 for sample in output])
        # shouldn't need to be! the whole batch should be input at once
        error = np.sum(.5 * (target - output)**2)
        self.errors.append(error)
        return error


    def _backpropagate(self, target, input):
        """
        Backpropagates the error through all the layers in one function call (as opposed to _feedforward which
        must be repeated)
        :param target:
        :return ??????????????????????????:
        """
        error = self._calculate_error(self.layer_outputs[-1], target)  # to record error
        print("ERROR", error)
        print("BACKPRAAAAAPPPPPPPPPPPPPPP")
        def last_layer_backprop(tgt):
            derivative_function = self._get_derivative(self.layer_activation_funcs[-1])
            delta = -np.dot(self.layer_logits[-2].T, np.multiply((tgt - self.layer_outputs[-1]), derivative_function(self.layer_logits[-1])))
            return delta
        def hidden_layer_backprop(tgt, inpt):
            deltas_backward = []
            for i in reversed(range(len(self.layer_logits) - 1)): # because we already did the last layer
                derivative_function = self._get_derivative(self.layer_activation_funcs[i])
                delta = (tgt - self.layer_outputs[-1])  # this increases in length completely right? so wouldn't it be better to do in in another loop, as parts?
                deltas_backward.append(delta)



        print("FERTIG")
        # self._descend_gradient(input, deltas_backwards, changes_backwards)


    def _descend_gradient(self, deltas, changes, GRAD):
        """
        Updates the weights of all layers with the DELTAS???? computed in the backprop step.
        :param deltas:
        :param changes:
        :param GRAD:
        """
        for layer in range(2, len(self.layer_sizes)):
            print("DESCENDING")
            epsilon = self.layer_epsilons[-layer]
            gradient = epsilon * GRAD[-layer]
            print(gradient)
            print(self.layer_weights[-layer])
            self.layer_weights[-layer] = self.layer_weights[-layer] - gradient
            print(self.layer_weights[-layer])
            # self.
        # deltas = list(reversed(deltas))
        # for l in range(len(self.layer_sizes)):
        #     print("LAYER", l)
        #     gradient = self.layer_epsilons[l] * deltas[l] * [self.layer_outputs[l-1] if l > 0 else input]
        #     print("GRADIENT:\n",gradient)
        #     print("GRADIENTshape:\n",gradient.shape)
        #     print(self.layer_weights[l])
        #     self.layer_weights[l] = self.layer_weights[l] - gradient
        #     print(self.layer_weights[l])
        # # self.layer_weights[layer]


    def learn(self, input, target, repitions):
        #PSEUDOCODE
        for r in repitions:
            self._feedforward(input)
            self._backpropagate(target, input)
            # self._descend_gradient(x,y,z)

    ##############---Training Functions---##############
    ################ Data Processing Functions ################

    def predict(self, input):
        """
        Propagates an input through the network to get the network's result, without doing the backprop step.
        :param input:
        :return prediction:
        """
        for layer in range(len(self.layer_sizes)):
            input = self._forward_step(layer, input)
        prediction = input  # just for clarification sake
        return prediction

    ##############---Data Processing Functions---##############
    ############## Save/Load Functions ##############
    def save(self, name=None, date=True):
        """
        Saves the current network structure and weights.  Can take in a specified name to save the file as, otherwise
        it is called "savednetwork".
        Can also append the date to the name.
        :param name:
        :param date:
        """
        gwt = self.global_weight_type
        ge = self.global_epsilon
        gaf = self.global_activation_func
        lw = self.layer_weights
        lwt = self.layer_weight_types
        laf = self.layer_activation_funcs
        le = self.layer_epsilons
        ls = self.layer_sizes
        network_packed = [gwt, ge, gaf, lw, lwt, laf, le, ls]

        if isinstance(name, str):
            if date:
                date = dt.today().isoformat()[:19]
                date = date.replace(':','-')
                f = open(name+date+'.pkl', 'wb')
                pickle.dump(network_packed, f, -1)
                f.close()
            else:
                f = open(name+'.pkl', 'wb')
                pickle.dump(network_packed, f, -1)
                f.close()
        else:
            if date:
                date = dt.today().isoformat()[:19]
                date = date.replace(':', '-')
                f = open('savednetwork'+date+'.pkl', 'wb')
                pickle.dump(network_packed, f, -1)
                f.close()
            else:
                f = open('savednetwork.pkl', 'wb')
                pickle.dump(network_packed, f, -1)
                f.close()


    def load(self, file):
        """
        Loads network structure and weights from a filename that was saved earlier with the save function.
        :param file:
        """
        f = open(file, 'rb')
        network_packed = pickle.load(f)
        f.close()

        self.global_weight_type = network_packed[0]
        self.global_epsilon = network_packed[1]
        self.global_activation_func = network_packed[2]
        self.layer_weights = network_packed[3]
        self.layer_weight_types = network_packed[4]
        self.layer_activation_funcs = network_packed[5]
        self.layer_epsilons = network_packed[6]
        self.layer_sizes = network_packed[7]

        ################---Save/Load Functions---################



if __name__ == "__main__":
    # inputs = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    #start debugging
    inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
    print(inputs)
    outputs = np.array([[0],[1],[1],[0]])
    print(outputs)
    NeuralNet = MultiLayerPerceptron('relu',.001,'ones')
    NeuralNet.create_layer(2, 2)
    NeuralNet.create_layer(2)
    NeuralNet.create_layer(1)
    # print(NeuralNet)
    print("DEBUG")
    # print(NeuralNet.layer_weights[0])
    first = NeuralNet.layer_weights
    input = inputs[0]
    for l in range(len(NeuralNet.layer_weights)):
        print("Stepping forward for layer:",l+1)
        input = NeuralNet._forward_step(l, input)
    # print(NeuralNet.layer_outputs)

    output = outputs[0]
    NeuralNet._backpropagate(inputs[0], output)
    last = NeuralNet.layer_weights

    print("\n\n\nFIRST{}\n\n\n\nLAST{}".format(first, last))
    #end debugging
    # # print(NeuralNet.layer_weight_types)
    # # print(NeuralNet.global_activation_func)
    # # print(NeuralNet.layer_activation_funcs)
    #
    # NeuralNet = MultiLayerPerceptron('relu',.001,'ones')
    # NeuralNet.create_layer(2, 2)
    # NeuralNet.create_layer(2)
    # NeuralNet.create_layer(1)
    # print("1:\n",NeuralNet)
    # NeuralNet.save('backup')
    #
    # NeuralNet2 = MultiLayerPerceptron('sigmoid',.0000001,'zeros')
    # print("2:\n",NeuralNet2)
    # NeuralNet2.load('savednetwork.pkl')
    # print("2LOAD:\n", NeuralNet2)

'''
So I need to make it impervious to batch size.  It should always work out through the matrix math
https://iamtrask.github.io/2015/07/12/basic-python-network/
'''