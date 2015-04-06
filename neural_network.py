#-*- coding: utf-8 -*-
u"""Contains classes and functions necessary to create, training and serialization backpropagation neural network."""

import random
import math
import pickle

def load(path):
    u"""Loads MultilayerPerceptron from file specified by path
    Args:
        path (str): Path to file.

    Returns:
        MultilayerPerceptron:
    """
    try:
        source = open(path, 'rb')
        try:
            return pickle.load(source)
        finally:
            source.close()
    except IOError:
        pass

def logistic(x):
    return 1.0 / (1.0 + math.exp(-x))

def logistic_derivative(output):
    return output * (1.0 - output)


class Neuron(object):
    def __init__(self, f, f_derivative):
        u"""
        Args:
            f (function): Activation function.
            f_derivative (function): Derivative of function f.
        """
        self.__f = f
        self.__f_derivative = f_derivative
        self.__inputs = []
        self.error = 0.0
        self.output = 0.0

    def reset_error(self):
        self.error = 0.0
        
    def add_input(self, neuron, weight = None):
        if weight is None:
            weight = random.uniform(-1, 1)
        self.__inputs.append(Synapse(neuron, weight))

    def add_inputs(self, layer):
        bias = Neuron(self.__f, self.__f_derivative)
        bias.output = 1.0
        self.add_input(bias)
        for neuron in layer.neurons:
            self.add_input(neuron)

    def feed_forward(self):
        u"""Computes response of single neuron"""
        output = 0.0
        for synapse in self.__inputs:
            output += synapse.weight * synapse.neuron.output
        self.output = self.__f(output)

    def back_propagate(self):
        u"""Modifies errors of all previous layer neurons connected to the current neuron"""
        for synapse in self.__inputs:
            synapse.neuron.error += self.error * synapse.weight

    def update_weights(self, learning_rate, momentum):
        for synapse in self.__inputs:
            tmp = synapse.weight
            synapse.weight += learning_rate * self.__f_derivative(self.output) * self.error * synapse.neuron.output + momentum * synapse.delta
            synapse.delta = synapse.weight - tmp


class Synapse(object):
    u"""Allows to link reference to neuron with the weight of synapse and other parameters."""
    def __init__(self, neuron, weight = None):
        u"""
        Args:
            neuron (Neuron):
            weight (float): Weight of the current synapse. Defaults to random float value between -1 and 1.
        """            
        self.neuron = neuron
        if weight is None:
            weight = random.uniform(-1, 1)
        self.weight = weight
        self.delta = 0 # Weight change in the previous iteration.


class NeuronLayer(object):
    def __init__(self, n_neurons, f, f_derivative):
        u"""
        Args:
            n_neurons (int): Number of neurons in this layer.
            f (function): Activation function.
            f_derivative (function): Derivative of function f.
        """
        self.neurons = [Neuron(f, f_derivative) for i in range(n_neurons)]

    def connect_to(self, layer):
        u"""Connects all neurons of current layer to all neurons of layer passed to the function as an argument.
        Args:
            layer (NeuronLayer):
        """
        for neuron in layer.neurons:
            neuron.add_inputs(self)

    def load_input(self, input_):
        u"""Initializes outputs of all neurons in current (input) layer with values from input_.
        Args:
            inputs (tuple):

        Raises:
            ValueError: If length of input_ is not equal to number of neurons in the layer.
        """
        if len(input_) != len(self.neurons):
            raise ValueError("Invalid output size. Current size: %d, expected: %d" % (len(input_), len(self.neurons)))
        for neuron, in_ in list(zip(self.neurons, input_)):
            neuron.output = in_

    def get_output(self):
        u"""
        Returns:
            list: Response of each neuron.
        """
        return [n.output for n in self.neurons]

    def feed_forward(self):
        for neuron in self.neurons:
            neuron.feed_forward()

    def back_propagate(self):
        for neuron in self.neurons:
            neuron.back_propagate()

    def compute_errors(self, expected_output):
        """Computes error of each neuron in the output layer. Then these errors can be propagated to the previous layers.

        Args:
            expected_output (tuple):

        Returns:
            NeuronLayer: Used to method chaining syntax.

        Raises:
            ValueError: If length of expected_output is not equal to number of neurons in the layer.
        """
        if len(expected_output) != len(self.neurons):
            raise ValueError("Invalid output size. Current size: %d, expected: %d" % (len(expected_output), len(self.neurons)))
        for neuron, out in list(zip(self.neurons, expected_output)):
            neuron.error = out - neuron.output
        return self

    def update_weights(self, learning_rate, momentum):
        for neuron in self.neurons:
            neuron.update_weights(learning_rate, momentum)
            neuron.reset_error()


class MultilayerPerceptron(object):
    def __init__(self, f, f_derivative, input_size, n_hidden_layers, hidden_size, output_size):
        u"""
        Args:
            f(function):
            f_derivative (function):
            input_size (int):
            n_hidden_layers (int):
            hidden_size (int):
            output_size (int):
        """
        self.input_layer = NeuronLayer(input_size, f, f_derivative)
        self.hidden_layers = [NeuronLayer(hidden_size, f, f_derivative) for i in range(n_hidden_layers)]
        self.output_layer = NeuronLayer(output_size, f, f_derivative)

        prev = self.hidden_layers[0]
        self.input_layer.connect_to(prev)
        for layer in self.hidden_layers[1:]:
            prev.connect_to(layer)
            prev = layer
        prev.connect_to(self.output_layer)

    def evaluate(self, input_):
        u"""Evaluates the perceptron.
        Args:
            input_ (tuple):

        Returns:
            list: Response of the neural network.

        Raises:
            ValueError: If length of input_ is not equal to number of neurons in the input layer.
        """
        if len(input_) != len(self.input_layer.neurons):
            raise ValueError("Invalid input size. Current size: %d, expected: %d" % (len(input_), len(self.__input_layer.neurons)))
        self.input_layer.load_input(input_)
        for layer in self.hidden_layers:
            layer.feed_forward()
        self.output_layer.feed_forward()
        return self.output_layer.get_output()

    def __getitem__(self, input_):
        u"""Allows to use short array syntax. For example: network[input_] has the same result as network.evaluate(input_)."""
        return self.evaluate(input_)

    def save(self, path):
        u"""The network can be serialized to file specified by path.
        Args:
            path (str): Path to file.
        """
        try:
            target = open(path, 'wb')
            try:
                pickle.dump(self, target)
            finally:
                target.close()
        except IOError:
            pass


class BackpropagationNetwork(MultilayerPerceptron, object):
    u"""Extends MultilayerPerceptron class. Adds features that allow us to train the network."""
    def __init__(self, input_size, n_hidden_layers, hidden_size, output_size, learning_rate = 0.9, momentum = 0.1, f = logistic, f_derivative = logistic_derivative):
        u"""
        Args:
            input_size (int):
            n_hidden_layers (int):
            hidden_size (int):
            output_size (int):
            learning_rate (float): Initial learning rate. Defaults to 0.9.
            momentum (float): Initial momentum. Defaults to 0.1.
            f (function): Activation function. Defaults to logistic function.
            f_derivative (function): Derivative of function f. Defaults to logistic function derivative.
        """
        MultilayerPerceptron.__init__(self, f, f_derivative, input_size, n_hidden_layers, hidden_size, output_size)
        self.__learning_rate = learning_rate
        self.__momentum = momentum

    def train(self, input_, expected_output):
        u"""Trains the network for the given data.
        Args:
            input_ (tuple):
            expected_output (tuple):
        """
        MultilayerPerceptron.evaluate(self, input_)
        self.output_layer.compute_errors(expected_output).back_propagate()
        for layer in self.hidden_layers:
            layer.back_propagate()

        self.output_layer.update_weights(self.__learning_rate, self.__momentum)
        for layer in self.hidden_layers:
            layer.update_weights(self.__learning_rate, self.__momentum)

    def __setitem__(self, input_, expected_output):
        u"""Allows to use short array syntax. For example: network[input_] = expected_output has the same result as network.train(input_, expected_output)."""
        self.train(input_, expected_output)
