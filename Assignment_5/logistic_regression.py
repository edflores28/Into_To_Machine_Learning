import random
import utilities


class LG:
    def __init__(self, train, test, learn_rate=0.01):
        self.train = train
        self.test = test
        self.weights = [random.uniform(-1.0, 1.0) for i in range(len(train[0]))]
        self.learn_rate = learn_rate

    def __predict_y(self, row):
        # Multiply the weights and each x value in the row
        # minus the classification value
        values = [a*b for a, b in zip(self.weights[1:], row[:-1])]
        summation = sum(values)
        return (utilities.sigmoid(summation + self.weights[0]))

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
            if y >= 0.5:
                correct += 1
            else:
                incorrect += 1
        # Return the values
        return correct, incorrect

    def train_model(self):
        prev_misclass = -100
        misclass = 0
        while misclass != prev_misclass:
            # Initial value for the delta weights
            deltas = [0.0 for i in range(len(self.weights))]
            # Iterate through each training entry
            for row in self.train:
                # Make a predication
                y = self.__predict_y(row)
                # Update the deltas
                for i in range(len(deltas)):
                    # Update w0
                    if i == 0:
                        deltas[0] += (1 - y)
                    # Update the other weights
                    else:
                        deltas[i] += (row[-1]-y)*row[i-1]
            # Update the weights
            for i in range(len(self.weights)):
                self.weights[i] += self.learn_rate*deltas[i]
            # Obtain the total missclassifications
            # and update the previous value
            prev_misclass = misclass
            unused, misclass = self.__determine_accuracy()

    def test_model(self):
        correct, incorrect = self.__determine_accuracy(False)
        print((correct*100)/(incorrect+correct))
