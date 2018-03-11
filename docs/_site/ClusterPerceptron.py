###################
# A cluster classification learning perceptron
# Good to use for understanding how a perceptron works.
# Displays decision boundary
# Author: John Berroa
###################

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal as multNorm


class ClusterPerceptron:
    """
    Creates a perceptron that learns to linearly separate two clusters
    """

    def __init__(self, data, epsilon=.0075):
        self.epsilon = epsilon # initialized with very low learning rate in order to visualize learning process
        self.dimensions = 2
        self.training_set = data[0]
        self.test_set = data[1]
        self.training_size = len(self.training_set)
        self.test_size = len(self.test_set)
        self.training_labels = data[2]
        self.test_labels = data[3]
        self.weights = np.random.random(self.dimensions + 1)  # +1 because of adding a bias


        print("Perceptron initialized with weights:\nW0 = {}     W1 = {}     W2 = {}"
              .format(round(self.weights[0],2), round(self.weights[1],2), round(self.weights[2],2)))


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


    def infer(self, datapoint):
        """
        Passes datapoint through the network to get an answer
        :param datapoint:
        :return output:  the answer of the inference
        """
        activation = np.dot(self.weights, datapoint[:-1])
        output = self.threshold(activation)
        return output


    def learn(self, output, label, datapoint):
        """
        Implements the perceptron learning rule
        :param output:
        :param label:
        :param datapoint:
        """
        delta_w = self.epsilon * ((label - output) * datapoint[:-1])
        self.weights += delta_w # perceptron learning rule


    def train(self, epochs=10):
        """
        Trains the perceptron for a certain amount of epochs, then tests it
        :param epochs:
        """
        for e in range(epochs):
            self.test(e, epochs)
            for i, step in enumerate(self.training_set):
                output = self.infer(step)
                self.learn(output, self.training_set[i][-1], step)
        self.test(epochs, epochs)


    def test(self, e, e2):
        """
        Tests the performance of the current weights against the test set and then prints the result
        :param e: these just pass through into the plotting function; they are the current epoch and epoch length
        :param e2: ^ see above
        :return:
        """
        print("Perceptron trained to weights:\nW0 = {}     W1 = {}     W2 = {}"
              .format(round(self.weights[0], 2), round(self.weights[1], 2), round(self.weights[2], 2)))
        results = []
        for i, step in enumerate(self.test_set):
            output = self.infer(step)
            if self.test_set[i][-1] == output:
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
        Plots the decision boundary and data for both the training and test datasets
        :param epoch: the current epoch
        :param epoch_length:  the total number of epochs
        :return:
        """
        plt.figure(1)
        plt.subplot(121)
        y_point = (0, (-self.weights[0] / self.weights[2]))
        x_point = ((-self.weights[0] / self.weights[1]), 0)
        try:
            slope = (y_point[1] - x_point[1]) / (y_point[0] - x_point[0]) # will not work if x and y intercepts are 0
        except ZeroDivisionError:
            print("X and Y intercepts are both zero.  Due to the way slope is calculated, this causes a division by zero.  Sorry.")
        y_out = lambda points: slope * points
        x = np.linspace(-10, 10, 100)
        plt.plot(x, y_out(x) + y_point[1], 'g--', linewidth=3, alpha=epoch / epoch_length if epoch < epoch_length else 1)

        # plot the training data to see how well the learning went
        if epoch == epoch_length: # plot each cluster in a different color
            for i in range(len(self.training_set)):
                if self.training_set[i][-1] == 0:
                    plt.scatter(self.training_set[i][1], self.training_set[i][2], color='red')
                elif self.training_set[i][-1] == 1:
                    plt.scatter(self.training_set[i][1], self.training_set[i][2], color='blue')
                else:
                    raise NotImplementedError("This condition should not happen; if it did, check your input data for problems")
            plt.ylim([-6, 6])
            plt.xlim([-6, 3])
            plt.title("Cluster Perceptron - Training Set")
            plt.xlabel("X")
            plt.ylabel("Y")

            # plot the test data to see if it generalizes
            plt.subplot(122)
            plt.plot(x, y_out(x) + y_point[1], 'g--', linewidth=3, alpha=epoch / epoch_length if epoch < epoch_length else 1)
            for i in range(len(self.test_set)):
                if self.test_set[i][-1] == 0:
                    plt.scatter(self.test_set[i][1], self.test_set[i][2], color='red')
                elif self.test_set[i][-1] == 1:
                    plt.scatter(self.test_set[i][1], self.test_set[i][2], color='blue')
                else:
                    raise NotImplementedError("This condition should not happen; if it did, check your input data for problems")
            plt.ylim([-6, 6])
            plt.xlim([-6, 3])
            plt.title("Cluster Perceptron - Test Set")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()


def generate_data(size):
    """
    Generates example data to test the perceptron.  However, any labeled linearly separable data will work.
    :return training_set, test_set, training_labels, test_labels:
    """
    # generate training and test sets, and put them into the plot
    training_set = np.vstack((multNorm([2,2],[[0.1, 0], [0, 1]],size), multNorm([-2,-4],[[1, 0], [0, 0.3]],size)))
    test_set = np.vstack((multNorm([2,2],[[0.1, 0], [0, 1]],size), multNorm([-2,-4],[[1, 0], [0, 0.3]],size)))

    # generate training and test labels, and bias
    training_labels = np.zeros(size)
    training_labels = np.concatenate((training_labels, np.ones(size)))
    test_labels = np.zeros(size)
    test_labels = np.concatenate((test_labels, np.ones(size)))
    bias = np.ones(size * 2) # because size is size of one cluster

    # add dimension, i.e. (200,) -> (200,1)
    training_labels = np.expand_dims(training_labels, axis=1)
    test_labels = np.expand_dims(test_labels, axis=1)
    bias = np.expand_dims(bias, axis=1)

    # concatenate labels onto data
    training_set = np.concatenate((training_set, training_labels), axis=1)
    test_set = np.concatenate((test_set, test_labels), axis=1)
    training_set = np.concatenate((bias, training_set), axis=1)
    test_set = np.concatenate((bias, test_set), axis=1)

    # shuffle so that it isn't ordered by label
    np.random.shuffle(training_set)
    np.random.shuffle(test_set)

    return training_set, test_set, training_labels, test_labels



if __name__ == "__main__":
    data_size = 100 # data is two times this size
    t,tst,tl,tstl = generate_data(data_size)
    data = (t, tst, tl, tstl)
    perceptron = ClusterPerceptron(data)
    perceptron.train(100)
