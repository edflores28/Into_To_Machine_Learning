import math
import random


class LR:
    def __init__(self, train, test, total_class, learn_rate=0.001):
        self.train = train
        self.test = test
        self.learn_rate = learn_rate
        self.weights = []
        self.total_class = total_class
        for i in range(total_class):
            temp = [random.uniform(-1.0, 1.0) for i in range(len(train[0]))]
            self.weights.append(temp)

    def __determine_accuracy(self, train=True):
        correct = 0
        incorrect = 0
        # Set the temp list according to the flag
        if train:
            temp = self.train
        else:
            temp = self.test
        # Iterate through each row and make a prediction
        for row in temp:
            y = self.__predict_y(row)
            # Choose the label with the highest value
            max_index = y.index(max(y))
            if max_index == row[-1]:
                correct += 1
            else:
                incorrect += 1
        # Return the values
        return correct, incorrect

    def __predict_y(self, row):
        y = []
        values = []
        # Iterate through all the class labels and calculate
        # the values
        for label in range(self.total_class):
            vals = [a*b for a, b in zip(self.weights[label][1:], row[:-1])]
            summation = sum(vals) + self.weights[label][0]
            values.append(summation)
        # Add all the values together
        total = sum(values)
        # Iterate through each class label and compute
        # the output
        for label in range(self.total_class):
            try:
                output = math.exp(values[label])/math.exp(total - values[label])
            except:
                output = 0.0
            y.append(output)
        return y

    def train_model(self):
        prev_misclass = -100
        misclass = 0
        while misclass != prev_misclass:
            class_deltas = []
            # Initalizez the deltas
            for i in range(self.total_class):
                class_deltas.append([0.0 for i in range(len(self.train[0]))])
            # Iterate through each row in the training set
            for row in self.train:
                # Get the list of predictions
                y = self.__predict_y(row)
                # For each class label calculate it's delta
                # vector
                for label in range(len(y)):
                    for weight in range(len(class_deltas[label])):
                        if weight == 0:
                            class_deltas[label][weight] += (1 - y[label])
                        else:
                            class_deltas[label][weight] += (row[-1]-y[label])*row[weight-1]
            # Update the weights
            for label in range(len(self.weights)):
                for i in range(len(self.weights[label])):
                    self.weights[label][i] += class_deltas[label][i] * self.learn_rate
            # Obtain the total missclassifications
            # and update the previous value
            prev_misclass = misclass
            unused, misclass = self.__determine_accuracy()
            if misclass == 0:
                break

    def test_model(self):
        correct, incorrect = self.__determine_accuracy(False)
        print((correct*100)/(incorrect+correct))
