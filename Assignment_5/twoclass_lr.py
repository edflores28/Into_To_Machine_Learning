import copy
import random
import utilities

'''
This class performs the two class logistic regression algorithm
'''

MAX_PRINT = 5


class Model:
    def __init__(self, train, test, learn_rate=0.001):
        '''
        Initialization
        '''
        self.train = train
        self.test = test
        self.weights = [random.uniform(-1.0, 1.0) for i in range(len(train[0]))]
        self.learn_rate = learn_rate
        print("Initial set of weights")
        print(self.weights[:MAX_PRINT])

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
        pred = 0
        count = 0
        # Set the temp list according to the flag
        if train:
            temp = self.train
        else:
            temp = self.test
        # Iterate through each row and make a prediction
        for row in temp:
            y = self.__predict_y(row)
            # Set the predition base on the value of y
            if y > 0.5:
                pred = 1
            else:
                pred = 0
            # Print to console
            if count < MAX_PRINT and not train:
                print("Predicted:", pred, "Expected:", row[-1])
                count += 1
            # Count the total of incorrect answers
            if pred != row[-1]:
                incorrect += 1
        # Return the values
        return incorrect

    def train_model(self):
        '''
        This method trains the two class logistic regression
        algorithm
        '''
        print("\nTraining two class logistic regression")
        prev_error = -100
        error = 0
        count = 0
        previous_weights = []
        first_iter = True
        while error != prev_error:
            # Initial value for the delta weights
            deltas = [0.0 for i in range(len(self.weights))]
            # Iterate through each training entry
            for row in self.train:
                # Make a predication
                y = self.__predict_y(row)
                if count < MAX_PRINT:
                    print("Predicted y =", y)
                    count += 1
                # Update the deltas
                for i in range(len(deltas)):
                    # Update w0
                    if i == 0:
                        deltas[0] += (1 - y)
                    # Update the other weights
                    else:
                        deltas[i] += (row[-1]-y)*row[i-1]
            # Save the weights before updating
            previous_weights = copy.deepcopy(self.weights)
            # Update the weights
            for i in range(len(self.weights)):
                self.weights[i] += self.learn_rate*deltas[i]
            print("Final set of weights")
            print(self.weights[:MAX_PRINT])
            # Obtain the total missclassifications
            # and update the previous value
            prev_error = error
            error = self.__determine_error()
            if error == 0:
                print("Error is zero no need to optimize")
                break
            if prev_error < error and not first_iter:
                print("The error increased this iteration, using previous weights")
                print("Previous Error", prev_error, "Current Error", error)
                self.weights = previous_weights
                break
            print("Previous Error", prev_error, "Current Error", error)
            print("Conintue optimizing weights")
            first_iter = False
        print("Finished regulating\n")

    def test_model(self):
        '''
        This method tests the trained model
        '''
        print("\nTesting twp class logistic regression")
        # Calculate the total error for the testing set
        incorrect = self.__determine_error(False)
        return((incorrect*100)/(len(self.test)))
