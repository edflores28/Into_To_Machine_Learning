import random
import utilities
import numpy as np
import itertools

class ZeroLayer:
    def __init__(self, train, test, total_classes, learn_rate=0.001):
        self.train = train
        self.test = test
        self.learn_rate = learn_rate
        self.total_classes = total_classes
        self.weightsO = []
        self.weightsI = []
        self.layer_two_weights = []
        # Create weights used by the output layer
        for i in range(total_classes):
            temp = [random.uniform(-1.0, 1.0) for i in range(len(train[0]))]
            self.weightsO.append(temp)

    def __forward_propagate(self, row):
        out_layer_values = []
        for label in range(self.total_classes):
            out_layer_values.append(utilities.calculate_sigmoid(self.weightsO[label], row[:-1]))
        return out_layer_values

    def __backpropagate(self, out_values, row):
        errors = [row[-1]-out_values[i] for i in range(len(out_values))]
        out_deltas = []
        temp_row = [1.0] + row[:-1]
        for label in range(self.total_classes):
            temp = [self.learn_rate*errors[label]*temp_row[i] for i in range(len(temp_row))]
            out_deltas.append(temp)
        return out_deltas

    def __weight_updates(self, updates):
        for label in range(self.total_classes):
            self.weightsO[label] = [self.weightsO[label][i] + updates[label][i] for i in range(len(self.weightsO[label]))]

    def train_model(self):
        for row in self.train:
            # Calculate each layer outputs
            out_values = self.__forward_propagate(row)
            output = self.__backpropagate(out_values, row)
            self.__weight_updates(output)

    def test_model(self):
        for row in self.test:
            output = self.__forward_propagate(row)
            print(output, row[-1])


class OneLayer:
    def __init__(self, train, test, hidden_nodes, total_classes, learn_rate=0.001):
        self.train = train
        self.test = test
        self.hidden_nodes = hidden_nodes
        self.learn_rate = learn_rate
        self.total_classes = total_classes
        self.weightsI = []
        self.weightsO = []
        # Create the weights for the hidden layer
        for i in range(self.hidden_nodes):
            temp = [random.uniform(-1.0, 1.0) for i in range(len(train[0]))]
            self.weightsI.append(temp)
        # Create the weights for the output later
        for i in range(total_classes):
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
            for out_node in range(self.total_classes):
                error += deltaO[out_node]*self.weightsO[out_node][node]
            deltaI.append(error * utilities.calculate_derivative(hidden_vals[node]))
        hidden_vals = [1] + hidden_vals
        for i in range(self.total_classes):
            temp = [self.learn_rate*hidden_vals[j]*deltaO[i] for j in range(len(hidden_vals))]
            self.weightsO[i] = [self.weightsO[i][j] + temp[j] for j in range(len(self.weightsO[i]))]

        for i in range(self.hidden_nodes):
            temp = [self.learn_rate*row[j]*deltaI[i] for j in range(len(row))]
            self.weightsI[i] = [self.weightsI[i][j] + temp[j] for j in range(len(self.weightsI[i]))]

    def __forward_propagate(self, row, test=False):
        row = [1.0] + row[:-1]
        hidden_outs = utilities.calculate_sigmoid_batch(self.weightsI, row, self.hidden_nodes)
        output = utilities.calculate_sigmoid_batch(self.weightsO, hidden_outs, self.total_classes)
        if test:
            return output
        return hidden_outs, output

    def train_model(self):
        row = self.train[0]
        for i in range(100):
            for row in range(len(self.train)):
                # Calculate each layer outputs
                input, output = self.__forward_propagate(self.train[row])
                #print(one_prop, out_prop)
                self.__backpropagate(output, input, self.train[row])
    def test_model(self):
        for row in self.test:
            output = self.__forward_propagate(row, True)
            print(output, row[-1])

class TwoLayer:
    def __init__(self, train, test, hidden_layers, hidden_nodes, total_classes, learn_rate=0.001):
        self.train = train
        self.test = test
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.learn_rate = learn_rate
        self.total_classes = total_classes
        self.weightsO = []
        self.weightsI = []
        self.layer_two_weights = []
        # Create the weights for the 1st hidden layer
        for i in range(hidden_nodes[0]):
            temp = [random.uniform(-1.0, 1.0) for i in range(len(train[0]))]
            self.weightsO.append(temp)
        # Create the weights for the 2nd hidden layer
        for i in range(hidden_nodes[1]):
            temp = [random.uniform(-1.0, 1.0) for i in range(hidden_nodes[0])]
            self.weightsI.append(temp)
        # Create the weights for the output later
        for i in range(total_classes):
            temp = [random.uniform(-1.0, 1.0) for i in range(hidden_nodes[1])]
            self.layer_two_weights.append(temp)

    def __forward_propagate(self, row):
        out_layer_values = []
        layer_one_values = []
        layer_two_values = []
        for node in range(self.hidden_nodes[0]):
            out_layer_values.append(utilities.calculate_sigmoid(self.weightsI[node], row[:-1]))

        for node in range(self.hidden_nodes[1]):
            out_layer_values.append(utilities.calculate_sigmoid(self.layer_two_weights[node], layer_one_values))

        for label in range(self.total_classes):
            out_layer_values.append(utilities.calculate_sigmoid(self.weightsO[label], layer_two_values))

        return layer_one_values, layer_two_values, out_layer_values

    def train_model(self):
        row = self.train[0]
        # Calculate each layer outputs
        layer_one, layer_two, layer_out = self.__forward_propagate(row)
        output = self.__backpropagate_no_layers(layer_out,row[-1])
        self.__weight_updates(output)
        # error = None
        # if self.hidden_layers == 0:
        #     error = [row[-1]-values[0][i] for i in range(len(values[0]))]
        # delta = None
        # delta = [self.learn_rate*error[0]*self.weights[0][i] for i in range(len(self.weights[0]))]
        # #update weights
        # self.weights[0] = [self.weights[0][i] + delta[i] for i in range(len(self.weights[0]))]
