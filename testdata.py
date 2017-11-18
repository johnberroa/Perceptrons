import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class ToyData:
    def __init__(self, dimensions, classes, size):
        self.dimensions = dimensions
        self.classes = classes
        self.size = size
        if self.dimensions == 1:
            # (b - a) * random_sample() + a to change range (does not include the upper number so they won't overlap)
            self.data = []  # in here because the other ones will be np arrays
            self.labels = []
            upper = 3
            lower = 1
            for i in range(classes):
                self.data.append(self.gen_random(upper, lower, self.size, i))
                self.labels.append([i + 1] * self.size)
            self.data = [item for sublist in self.data for item in sublist]
            self.labels = [item for sublist in self.labels for item in sublist]
        elif self.dimensions == 2:
            self.data = np.zeros((self.classes, self.size, 2))  # 2 for xy values
            self.labels = np.ones(self.size)
            upper = 3
            lower = 1
            for i in range(self.classes - 1):
                l2 = np.ones(self.size) * i + 2
                self.labels = np.vstack((self.labels, l2))
            for i in range(self.classes):
                for p_x in range(self.size):
                    self.data[i][p_x][0] = self.gen_random(upper, lower, 1, i)
            for i in range(self.classes):
                for p_y in range(self.size):
                    self.data[i][p_y][1] = self.gen_random(upper, lower, 1, i)
        elif self.dimensions == 3:
            self.data = np.zeros((self.classes, self.size, 3))  # 2 for xy values
            self.labels = np.ones(self.size)
            upper = 3
            lower = 1
            for i in range(self.classes - 1):
                l2 = np.ones(self.size) * i + 2
                self.labels = np.vstack((self.labels, l2))
            for i in range(self.classes):
                for p_x in range(self.size):
                    self.data[i][p_x][0] = self.gen_random(upper, lower, 1, i)
            for i in range(self.classes):
                for p_y in range(self.size):
                    self.data[i][p_y][1] = self.gen_random(upper, lower, 1, i)
            for i in range(self.classes):
                for p_z in range(self.size):
                    self.data[i][p_z][2] = self.gen_random(upper, lower, 1, i)
        else:
            raise ValueError(
                'Can only generate toy data up to 3 dimensions.  You input: {} dimensions'.format(dimensions))


    def gen_random(self, upper, lower, size, class_i):
        return ((upper + (class_i * 2 + 1)) - (lower + (class_i * 2 + 1))) * np.random.random(size) + (
        lower + (class_i * 2 + 1))


    def plot(self, prediction_model=False):
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
        if not isinstance(prediction_model, bool):
            print("Plotting: Colors based on predictions of class")
            if self.dimensions == 1:
                counter = 0
                color_picker = -1  # it will uptick upon starting the loop
                for i in range(len(self.data)):
                    if counter % self.size == 0:
                        color_picker += 1
                    if color_picker > 5:
                        color_picker = 0
                    color = color_picker
                    if prediction_model.predict(i) != self.labels[i]:
                        color = prediction_model.predict(i) - 1  # because 0 indexing
                    plt.scatter(self.data[i], 0, color=colors[color])
                    counter += 1
                plt.show()
            elif self.dimensions == 2:
                raise NotImplementedError
            elif self.dimensions == 3:
                raise NotImplementedError
            else:
                raise ValueError(
                    'Can only generate toy data up to 3 dimensions.  You input: {} dimensions'.format(self.dimensions))
        else:
            print("Plotting: Colors based on class")
            if self.dimensions == 1:
                counter = 0
                color_picker = -1  # it will uptick upon starting the loop
                for i in range(len(self.data)):
                    if counter % self.size == 0:
                        color_picker += 1
                    if color_picker > 5:
                        color_picker = 0
                    plt.scatter(self.data[i], 0, color=colors[color_picker])
                    counter += 1
                plt.show()
            elif self.dimensions == 2:
                color_picker = 0
                for c in range(self.classes):
                    for p in range(self.size):
                        plt.scatter(self.data[c][p][0], self.data[c][p][1], color=colors[color_picker])
                    color_picker += 1
                    if color_picker > 5:
                        color_picker = 0
                plt.show()
            elif self.dimensions == 3:
                color_picker = 0
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                for c in range(self.classes):
                    for p in range(self.size):
                        ax.scatter(self.data[c][p][0], self.data[c][p][1], self.data[c][p][2], c=colors[color_picker])
                    color_picker += 1
                    if color_picker > 5:
                        color_picker = 0
                plt.show()
            else:
                raise ValueError(
                    'Can only generate toy data up to 3 dimensions.  You input: {} dimensions'.format(self.dimensions))
