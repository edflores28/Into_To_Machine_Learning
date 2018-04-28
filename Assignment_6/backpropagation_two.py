import random
import utilities
import copy

MAX_EPOCH = 1000


class Model:
    def __init__(self, train, test, hidden_nodes, outputs, learn_rate=0.001):
        '''
        Initialization
        '''
        self.train = train
        self.test = test
        self.validation = []
        # Pull 10% of the training set to use for early stopping
        for i in range(int(len(train)*.10)):
            self.validation.append(self.train.pop())
        self.hidden_nodes = hidden_nodes
        self.learn_rate = learn_rate
        self.outputs = outputs
        self.weightsI = []
        self.weightsH = []
        self.weightsO = []
        # Create the weights for input -> hidden layer 1 including the bias
        for i in range(self.hidden_nodes[0]):
            temp = [random.uniform(-1.0, 1.0) for i in range(len(train[0]))]
            self.weightsI.append(temp)
        # Create the weights for the hidden layer 1 -> hidden layer 2 including the bias
        for i in range(self.hidden_nodes[1]):
            temp = [random.uniform(-1.0, 1.0) for i in range(hidden_nodes[0]+1)]
            self.weightsH.append(temp)
        # Create the weights for the hidden layer 2 -> output later including the bias
        for i in range(outputs):
            temp = [random.uniform(-1.0, 1.0) for i in range(hidden_nodes[1]+1)]
            self.weightsO.append(temp)

    def __backpropagate(self, row):
        '''
        This method propagates the error from the outputs back
        to the inputs and performs weight updates
        '''
        deltaO = []
        deltaH = []
        deltaI = []
        # Add a 1 to the row and hidden values
        # to account for the bias weight
        t_row = [1] + row[:-1]
        h2_vals = [1] + self.h2_outs
        h1_vals = [1] + self.h1_outs
        # Create a list for expected values
        expected = utilities.create_expected(self.outputs, row)
        # Iterate through the outputs and calculate the
        # delta values
        for i in range(len(self.output)):
            error = expected[i] - self.output[i]
            deltaO.append(utilities.calculate_derivative(self.output[i]) * error)
        # Iterate through the hidden nodes and calculate
        # the delta values
        for node in range(self.hidden_nodes[1]):
            error = 0.0
            for out_node in range(self.outputs):
                error += deltaO[out_node]*self.weightsO[out_node][node]
            deltaH.append(error * utilities.calculate_derivative(self.h2_outs[node]))
        # Iterate through the hidden nodes and calculate
        # the delta values
        for node in range(self.hidden_nodes[0]):
            error = 0.0
            for h_node in range(self.hidden_nodes[1]):
                error += deltaH[out_node]*self.weightsH[out_node][node]
            deltaI.append(error * utilities.calculate_derivative(self.h1_outs[node]))
        # Iterate through the output nodes and
        # update their weights
        for i in range(self.outputs):
            temp = [self.learn_rate*h2_vals[j]*deltaO[i] for j in range(len(h2_vals))]
            self.weightsO[i] = [self.weightsO[i][j] + temp[j] for j in range(len(self.weightsO[i]))]
        # Iterate through the layer 2 hidden nodes and
        # update their weights
        for i in range(self.hidden_nodes[1]):
            temp = [self.learn_rate*h1_vals[j]*deltaH[i] for j in range(len(h1_vals))]
            self.weightsH[i] = [self.weightsH[i][j] + temp[j] for j in range(len(self.weightsH[i]))]
        # Iterate through the layer 1 hidden nodes and
        # update their weights
        for i in range(self.hidden_nodes[0]):
            temp = [self.learn_rate*t_row[j]*deltaI[i] for j in range(len(t_row))]
            self.weightsI[i] = [self.weightsI[i][j] + temp[j] for j in range(len(self.weightsI[i]))]

    def __forward_propagate(self, row):
        '''
        This method calculates all the node outputs for each
        layer in the network
        '''
        # Append a 1 to the row to account for the bias
        row = [1.0] + row[:-1]
        self.h1_outs = utilities.calculate_sigmoid_batch(self.weightsI, row, self.hidden_nodes[0])
        self.h2_outs = utilities.calculate_sigmoid_batch(self.weightsH, self.h1_outs, self.hidden_nodes[1])
        self.output = utilities.calculate_sigmoid_batch(self.weightsO, self.h2_outs, self.outputs)

    def train_model(self):
        prev_weightsO = []
        prev_weightsH = []
        prev_weightsI = []
        prev_correct = 0
        epoch = 0
        # Train the network until broken
        while True:
            # Iterate through all the rows
            for row in range(len(self.train)):
                # Calculate the outputs of each layer
                self.__forward_propagate(self.train[row])
                # Backpropagate the errors
                self.__backpropagate(self.train[row])
            # Determine the classification accuracy of the network
            correct = self.test_model(True)
            # If the correct value is less than previous
            # correct value or the epoch is greater than
            # the maximum epoch break the loop.
            if correct < prev_correct or epoch > MAX_EPOCH:
                # Restore the weights
                self.weightsI = prev_weightsI
                self.weightsH = prev_weightsH
                self.weightsO = prev_weightsO
                print(prev_correct, correct)
                break
            # Save the weights, the correct value
            # and increment the epoch
            prev_correct = correct
            prev_weightsI = copy.deepcopy(self.weightsI)
            prev_weightsH = copy.deepcopy(self.weightsH)
            prev_weightsO = copy.deepcopy(self.weightsO)
            epoch += 1

    def __predict(self, row):
        '''
        This method predicts the classification of the row
        '''
        self.__forward_propagate(row)
        return utilities.network_predict(self.output)

    def test_model(self, validate=False):
        '''
        This method tests the network
        '''
        correct = 0
        incorrect = 0
        test = self.test
        # Use the validation set if the flag is set
        if validate:
            test = self.validation
        # Iterate over each row
        for row in test:
            pred = self.__predict(row)
            if pred == row[-1]:
                correct += 1
            else:
                incorrect += 1
        return (correct*100)/(correct+incorrect)
