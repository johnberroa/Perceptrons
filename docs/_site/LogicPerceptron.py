###################
# A logical operator learning perceptron
# Good to use for understanding how a perceptron works.
# Displays decision boundary
# Author: John Berroa
###################

import numpy as np
import matplotlib.pyplot as plt


class LogicPerceptron:
    """
    Creates a perceptron that learns logical operators AND, OR, NAND, and NOR
    """

    def __init__(self, epsilon=.0075, training_size=100, test_size=100):
        self.epsilon = epsilon # initialized with very low learning rate in order to visualize learning process
        self.dimensions = 2
        self.training_size = training_size
        self.test_size = test_size
        self.weights = np.random.random(self.dimensions + 1)  # +1 because of adding a bias
        self.plot_points = [[0,0],[0,1],[1,0],[1,1]]
        self.plot_colors = []

        print("Perceptron initialized with weights:\nW0 = {}     W1 = {}     W2 = {}"
              .format(round(self.weights[0],2), round(self.weights[1],2), round(self.weights[2],2)))


    def generate_datasets(self):
        """
        Generates training and test data sets as 1s and 0s with 1s as the final column for bias
        :return training_set, test_set:
        """
        # generate bias
        bias = np.ones(self.training_size)  # shape (100,)
        bias = np.expand_dims(bias, axis=1)  # shape (100,1)

        # generate training set
        training_set = np.random.randint(2, size=(self.training_size, self.dimensions))  # pairs of 1s and 0s
        training_set = np.concatenate((bias, training_set), axis=1)

        # generate test set
        test_set = np.random.randint(2, size=(self.test_size, self.dimensions))  # pairs of 1s and 0s
        test_set = np.concatenate((bias, test_set), axis=1)
        return training_set, test_set


    def generate_labels(self, function, dataset):
        """
        Passes through the datapoints to get the correct classification based on the logical function provided
        :param function:
        :param dataset:
        :return labels:
        """
        labels = []
        for datapoint in dataset:
            labels.append(function(datapoint[1], datapoint[2]))
        return labels


    def threshold(self, activation):
        """
        Simple step function threshold
        :param activation:
        :return 1 or 0:
        """
        if activation >= 0:
            return 1
        else:
            return 0


    def functions_to_learn(self, selector):
        """
        Functional definitions for the perceptron to learn
        Instantiates plots for visualization of the decision boundary
        :param selector: selects which function to activate
        :return function:
        """
        if selector == 'and':
            function = lambda x1, x2: x1 and x2
            for point in self.plot_points:
                self.plot_colors.append(function(point[0], point[1]))
            for color, point in enumerate(self.plot_points):
                plt.scatter(*point, s=50, c='b' if self.plot_colors[color] == 1 else 'r')
            print("Perceptron will now learn '{}'...\n\n".format(selector))
            return function
        elif selector == 'or':
            function = lambda x1, x2: x1 or x2
            for point in self.plot_points:
                self.plot_colors.append(function(point[0], point[1]))
            for color, point in enumerate(self.plot_points):
                plt.scatter(*point, s=50, c='b' if self.plot_colors[color] == 1 else 'r')
            print("Perceptron will now learn '{}'...\n\n".format(selector))
            return function
        elif selector == 'nand':
            function = lambda x1, x2: not (x1 and x2)
            for point in self.plot_points:
                self.plot_colors.append(function(point[0], point[1]))
            for color, point in enumerate(self.plot_points):
                plt.scatter(*point, s=50, c='b' if self.plot_colors[color] == 1 else 'r')
            print("Perceptron will now learn '{}'...\n\n".format(selector))
            return function
        elif selector == 'nor':
            function = lambda x1, x2: not (x1 or x2)
            for point in self.plot_points:
                self.plot_colors.append(function(point[0], point[1]))
            for color, point in enumerate(self.plot_points):
                plt.scatter(*point, s=50, c='b' if self.plot_colors[color] == 1 else 'r')
            print("Perceptron will now learn '{}'...\n\n".format(selector))
            return function
        else:
            raise ValueError("Incorrect function to learn.  Pick and/or/nand/nor.  Input was:", selector)


    def infer(self, datapoint):
        """
        Passes datapoint through the network to get an answer
        :param datapoint:
        :return output:  the answer of the inference
        """
        activation = np.dot(self.weights, datapoint)
        output = self.threshold(activation)
        return output


    def learn(self, output, label, datapoint):
        """
        Implements the perceptron learning rule
        :param output:
        :param label:
        :param datapoint:
        """
        delta_w = self.epsilon * ((label - output) * datapoint)
        self.weights += delta_w # perceptron learning rule


    def train(self, function_string, epochs):
        """
        Trains the perceptron for a certain amount of epochs, then tests it
        :param function_string:
        :param epochs:
        """
        training_set, test_set = self.generate_datasets()
        function = self.functions_to_learn(function_string)
        labels = self.generate_labels(function, training_set)
        for e in range(epochs):
            self.test(function, test_set, e, epochs)
            for i, step in enumerate(training_set):
                output = self.infer(step)
                self.learn(output, labels[i], step)
        self.test(function, test_set, epochs, epochs)


    def test(self, function, test_set, e, e2):
        """
        Tests the performance of the current weights against the test set and then prints the result
        :param function:
        :param test_set:
        :param e: these just pass through into the plotting function; they are the current epoch and epoch length
        :param e2: ^ see above
        :return:
        """
        print("Perceptron trained to weights:\nW0 = {}     W1 = {}     W2 = {}"
              .format(round(self.weights[0], 2), round(self.weights[1], 2), round(self.weights[2], 2)))
        labels = self.generate_labels(function, test_set)
        results = []
        for i, step in enumerate(test_set):
            output = self.infer(step)
            if labels[i] == output:
                results.append(1)
            else:
                results.append(0)
        correct_results = np.count_nonzero(results)
        performance = correct_results / self.test_size
        print("Test performance after training: {}%\n----".format(performance * 100))
        self.plot_decision_boundary(e, e2)
        return performance


    def plot_decision_boundary(self, epoch, epoch_length):
        """
        Plots the decision boundary and then shows the plot
        :param epoch: the current epoch
        :param epoch_length:  the total number of epochs
        :return:
        """
        y_point = (0, (-self.weights[0] / self.weights[2]))
        x_point = ((-self.weights[0] / self.weights[1]), 0)
        try:
            slope = (y_point[1] - x_point[1]) / (y_point[0] - x_point[0]) # will not work if x and y intercepts are 0
        except ZeroDivisionError:
            print("X and Y intercepts are both zero.  Due to the way slope is calculated, this causes a division by zero.  Sorry.")
        y_out = lambda points: slope * points
        x = np.linspace(-10, 10, 100)
        plt.plot(x, y_out(x) + y_point[1], 'g--', linewidth=3, alpha=epoch/epoch_length + .2 if epoch < epoch_length else 1)
        if epoch == epoch_length:
            plt.ylim([-.2, 1.2])
            plt.xlim([-.2, 1.2])
            plt.title("Logic Perceptron")
            plt.xlabel("True(1) or False(0)")
            plt.ylabel("True(1) or False(0)")
            plt.show()





if __name__ == "__main__":
    perceptron = LogicPerceptron()
    perceptron.train('and', 10) # can do 'and', 'or', 'nand', and 'nor'

