import random
import utilities

'''
This class performs the two class logistic regression algorithm
'''


class Model:
    def __init__(self, train, test, learn_rate=0.001):
        '''
        Initialization
        '''
        self.train = train
        self.test = test
        self.weights = [random.uniform(-1.0, 1.0) for i in range(len(train[0]))]
        self.learn_rate = learn_rate

    def __predict_y(self, row):
        '''
        This method makes a prediction for the given row
        '''
        # Multiply the weights and each x value in the row
        # minus the classification value
        values = [a*b for a, b in zip(self.weights[1:], row[:-1])]
        summation = sum(values)
        return (utilities.sigmoid(summation + self.weights[0]))

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
            if y < 0.5:
                incorrect += 1
        # Return the values
        return incorrect

    def train_model(self):
        '''
        This method trains the two class logistic regression
        algorithm
        '''
        prev_error = -100
        error = 0
        while error != prev_error:
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
            prev_error = error
            error = self.__determine_error()

    def test_model(self):
        '''
        This method tests the trained model
        '''
        # Calculate the total error for the testing set
        incorrect = self.__determine_error(False)
        return((incorrect*100)/(len(self.test)))
