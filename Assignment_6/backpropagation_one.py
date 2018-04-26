import random
import utilities
import copy

class Model:
    def __init__(self, train, test, hidden_nodes, outputs, learn_rate=0.001):
        self.train = train
        self.test = test
        self.validation = []
        for i in range(int(len(train)*.10)):
            self.validation.append(self.train.pop())
        self.hidden_nodes = hidden_nodes
        self.learn_rate = learn_rate
        self.outputs = outputs
        self.weightsI = []
        self.weightsO = []
        # Create the weights for the hidden layer
        for i in range(self.hidden_nodes):
            temp = [random.uniform(-1.0, 1.0) for i in range(len(train[0]))]
            self.weightsI.append(temp)
        # Create the weights for the output later
        for i in range(outputs):
            temp = [random.uniform(-1.0, 1.0) for i in range(hidden_nodes+1)]
            self.weightsO.append(temp)

    def __backpropagate(self, out_vals, hidden_vals, row):
        deltaO = []
        row = [1] + row[:-1]
        for i in range(len(out_vals)):
            error = row[-1] - out_vals[i]
            deltaO.append(utilities.calculate_derivative(out_vals[i]) * error)

        deltaI = []
        for node in range(self.hidden_nodes):
            error = 0.0
            for out_node in range(self.outputs):
                error += deltaO[out_node]*self.weightsO[out_node][node]
            deltaI.append(error * utilities.calculate_derivative(hidden_vals[node]))
        hidden_vals = [1] + hidden_vals
        for i in range(self.outputs):
            temp = [self.learn_rate*hidden_vals[j]*deltaO[i] for j in range(len(hidden_vals))]
            self.weightsO[i] = [self.weightsO[i][j] + temp[j] for j in range(len(self.weightsO[i]))]

        for i in range(self.hidden_nodes):
            temp = [self.learn_rate*row[j]*deltaI[i] for j in range(len(row))]
            self.weightsI[i] = [self.weightsI[i][j] + temp[j] for j in range(len(self.weightsI[i]))]

    def __forward_propagate(self, row, test=False):
        row = [1.0] + row[:-1]
        hidden_outs = utilities.calculate_sigmoid_batch(self.weightsI, row, self.hidden_nodes)
        output = utilities.calculate_sigmoid_batch(self.weightsO, hidden_outs, self.outputs)
        if test:
            return output
        return hidden_outs, output

    def train_model(self):
        prev_weightsO = []
        prev_weightsI = []
        prev_correct = 0
        while True:
            # Iterate through all the rows
            for row in range(len(self.train)):
                # Calculate the outputs of each layer
                input, output = self.__forward_propagate(self.train[row])
                # Backpropagate the errors
                self.__backpropagate(output, input, self.train[row])
            correct = self.test_model(True);
            print(correct)
            if correct < prev_correct:
                self.weightsI = prev_weightsI
                self.weightsO = prev_weightsO
                break

            prev_correct = correct
            prev_weightsI = copy.deepcopy(self.weightsI)
            prev_weightsO = copy.deepcopy(self.weightsO)

    def __predict(self, row):
        output = self.__forward_propagate(row, True)
        if self.outputs == 1:
            if output[0] > 0.5:
                return 1
            else:
                return 0


    def test_model(self, validate=False):
        correct = 0
        incorrect = 0
        A = self.test
        if validate:
            A = self.validation

        for row in A:
            pred = self.__predict(row)
            if pred == row[-1]:
                correct += 1
            else:
                incorrect += 1
        return (correct*100)/(correct+incorrect)
