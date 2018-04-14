import math
import random

'''
This class performs the multi class logistic regression algorithm
'''


class Model:
    def __init__(self, train, test, total_class, learn_rate=0.001):
        '''
        Initialization
        '''
        self.train = train
        self.test = test
        self.learn_rate = learn_rate
        self.weights = []
        self.total_class = total_class
        for i in range(total_class):
            temp = [random.uniform(-1.0, 1.0) for i in range(len(train[0]))]
            self.weights.append(temp)

    def __determine_error(self, train=True):
        '''
        This method determines the total error of the
        set. If train is set to true the training set
        will be used otherwise the testing set is used
        '''
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
            if max_index != row[-1]:
                incorrect += 1
        # Return the values
        return incorrect

    def __predict_y(self, row):
        '''
        This method makes a prediction for the given row
        '''
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
        '''
        This method trains the two class logistic regression
        algorithm
        '''
        prev_error = -100
        error = 0
        while error != prev_error:
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
            prev_error = error
            error = self.__determine_error()
            if error == 0:
                break

    def test_model(self):
        '''
        This method tests the trained model
        '''        
        # Calculate the total error for the testing set
        incorrect = self.__determine_error(False)
        return((incorrect*100)/(len(self.test)))
