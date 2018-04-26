import random
import utilities
import copy

class TwoLayer:
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
        self.weightsH = []
        self.weightsO = []
        # Create the weights for input -> hidden layer 1
        for i in range(self.hidden_nodes[0]):
            temp = [random.uniform(-1.0, 1.0) for i in range(len(train[0]))]
            self.weightsI.append(temp)
        # Create the weights for the hidden layer 1 -> hidden layer 2
        for i in range(self.hidden_nodes[1]):
            temp = [random.uniform(-1.0, 1.0) for i in range(hidden_nodes[0]+1)]
            self.weightsH.append(temp)
        # Create the weights for the hidden layer 2 -> output later
        for i in range(outputs):
            temp = [random.uniform(-1.0, 1.0) for i in range(hidden_nodes[1]+1)]
            self.weightsO.append(temp)

    def __backpropagate(self, row):
        deltaO = []
        row = [1] + row[:-1]
        for i in range(len(self.output)):
            error = row[-1] - self.output[i]
            deltaO.append(utilities.calculate_derivative(self.output[i]) * error)

        deltaH =[]
        for node in range(self.hidden_nodes[1]):
            error = 0.0
            for out_node in range(self.outputs):
                error += deltaO[out_node]*self.weightsO[out_node][node]
            deltaH.append(error * utilities.calculate_derivative(self.h2_outs[node]))

        deltaI = []
        for node in range(self.hidden_nodes[0]):
            error = 0.0
            for h_node in range(self.hidden_nodes[1]):
                error += deltaH[out_node]*self.weightsH[out_node][node]
            deltaI.append(error * utilities.calculate_derivative(self.h1_outs[node]))

        h2_vals = [1] + self.h2_outs
        h1_vals = [1] + self.h1_outs

        for i in range(self.outputs):
            temp = [self.learn_rate*h2_vals[j]*deltaO[i] for j in range(len(h2_vals))]
            self.weightsO[i] = [self.weightsO[i][j] + temp[j] for j in range(len(self.weightsO[i]))]

        for i in range(self.outputs):
            temp = [self.learn_rate*h2_vals[j]*deltaO[i] for j in range(len(h2_vals))]
            self.weightsO[i] = [self.weightsO[i][j] + temp[j] for j in range(len(self.weightsO[i]))]

        for i in range(self.hidden_nodes[1]):
            temp = [self.learn_rate*row[j]*deltaI[i] for j in range(len(row))]
            self.weightsI[i] = [self.weightsI[i][j] + temp[j] for j in range(len(self.weightsI[i]))]

    def __forward_propagate(self, row, test=False):
        row = [1.0] + row[:-1]
        self.h1_outs = utilities.calculate_sigmoid_batch(self.weightsI, row, self.hidden_nodes[0])
        self.h2_outs = utilities.calculate_sigmoid_batch(self.weightsH, self.h1_outs, self.hidden_nodes[1])
        self.output = utilities.calculate_sigmoid_batch(self.weightsO, self.h2_outs, self.outputs)

    def train_model(self):
        prev_weightsO = []
        prev_weightsH = []
        prev_weightsI = []
        prev_correct = 0
        while True:
            # Iterate through all the rows
            for row in range(len(self.train)):
                # Calculate the outputs of each layer
                self.__forward_propagate(self.train[row])
                # Backpropagate the errors
                self.__backpropagate(self.train[row])
            correct = self.test_model(True);
            print(correct)
            if correct < prev_correct:
                self.weightsI = prev_weightsI
                self.weightsH = prev_weightsH
                self.weightsO = prev_weightsO
                break

            prev_correct = correct
            prev_weightsI = copy.deepcopy(self.weightsI)
            prev_weightsH = copy.deepcopy(self.weightsH)
            prev_weightsO = copy.deepcopy(self.weightsO)

    def __predict(self, row):
        self.__forward_propagate(row, True)
        if self.outputs == 1:
            if self.output[0] > 0.5:
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
