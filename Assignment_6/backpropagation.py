import random
import utilities

class ZeroLayers:
    def __init__(self, train, test, total_classes, learn_rate=0.001):
        self.train = train
        self.test = test
        self.learn_rate = learn_rate
        self.total_classes = total_classes
        self.out_weights = []
        self.layer_one_weights = []
        self.layer_two_weights = []
        # Create weights used by the output layer
        for i in range(total_classes):
            temp = [random.uniform(-1.0, 1.0) for i in range(len(train[0]))]
            self.out_weights.append(temp)

    def __calculate_sigmoid(self, weights, input):
        '''
        This method multplies the weights and input
        and calculates the sigmoid
        '''
        values = [a*b for a, b in zip(weights[1:], input)]
        return utilities.sigmoid(sum(values) + weights[0])

    def __forward_propagate(self, row):
        out_layer_values = []
        for label in range(self.total_classes):
            out_layer_values.append(utilities.calculate_sigmoid(self.out_weights[label], row[:-1]))
        return out_layer_values

    def __backpropagate(self, out_values, row):
        errors = [row[-1]-out_values[i] for i in range(len(out_values))]
        out_deltas = []
        for label in range(self.total_classes):
            temp = [self.learn_rate*errors[label]*row[:-1] for i in range(len(row[:-1]))]
            out_deltas.append(temp)
        return out_deltas

    def __weight_updates(self, updates):
        for label in range(self.total_classes):
            self.out_weights[label] = [self.out_weights[label][i] + updates[label][i] for i in range(len(self.out_weights[label]))]

    def train_model(self):
        row = self.train[0]
        # Calculate each layer outputs
        out_values = self.__forward_propagate(row)
        output = self.__backpropagate_no_layers(out_values, row)
        print(self.out_weights[0])
        self.__weight_updates(output)
        print(self.out_weights[0])
        # error = None
        # if self.hidden_layers == 0:
        #     error = [row[-1]-values[0][i] for i in range(len(values[0]))]
        # delta = None
        # delta = [self.learn_rate*error[0]*self.weights[0][i] for i in range(len(self.weights[0]))]
        # #update weights
        # self.weights[0] = [self.weights[0][i] + delta[i] for i in range(len(self.weights[0]))]

class OneLayer:
    def __init__(self, train, test, hidden_nodes, total_classes, learn_rate=0.001):
        self.train = train
        self.test = test
        self.hidden_nodes = hidden_nodes
        self.learn_rate = learn_rate
        self.total_classes = total_classes
        self.out_weights = []
        self.layer_one_weights = []
        # Create the weights for the hidden layer
        for i in range(hidden_nodes):
            temp = [random.uniform(-1.0, 1.0) for i in range(len(train[0]))]
            self.layer_one_weights.append(temp)
        # Create the weights for the output later
        for i in range(total_classes):
            temp = [random.uniform(-1.0, 1.0) for i in range(hidden_nodes)]
            self.out_weights.append(temp)

    def __backpropagate(self, out_values, layer_one_values, row):
        errors = [row[-1]-out_values[i] for i in range(len(out_values))]
        out_deltas = []
        layer_one = []
        for label in range(self.total_classes):
            temp = [self.learn_rate*errors[label]*layer_one_values[i] for i in range(len(layer_one_values))]
            out_deltas.append(temp)

        for node in range(self.hidden_nodes):
            total = 0.0
            for label in range(self.total_classes):
                total += errors[label] * self.out_weights[label][node]
            print(self.hidden_nodes)
            print(self.layer_one_weights[node][0])
            value = [total * utilities.calculate_derivative(self.layer_one_weights[node][i]) for i in range(len(self.layer_one_weights[node]))]
            #value = total * utilities.calculate_derivative(self.layer_one_weights[node])
            temp = [value[i] * row[:-1][i] for i in range(len(row[:-1]))]
            layer_one.append(temp)
        return out_deltas, layer_one

    def __forward_propagate(self, row):
        out_layer_values = []
        layer_one_values = []
        for node in range(self.hidden_nodes):
            layer_one_values.append(utilities.calculate_sigmoid(self.layer_one_weights[node], row[:-1]))
        for label in range(self.total_classes):
            out_layer_values.append(utilities.calculate_sigmoid(self.out_weights[label], layer_one_values))
        return layer_one_values, out_layer_values

    def __weight_updates(self, output, layer):
        for label in range(self.total_classes):
            temp = [self.out_weights[label][i] + output[label][i] for i in range(len(self.out_weights[label]))]
            self.out_weights[label] = temp
        for node in range(self.hidden_nodes):
            temp = [self.layer_one_weights[label][i] + layer[node][i] for i in range(len(self.layer_one_weights[label]))]
            self.layer_one_weights[label] = temp

    def train_model(self):
        row = self.train[0]
        # Calculate each layer outputs
        layer_one, layer_out = self.__forward_propagate(row)
        output, layer_one = self.__backpropagate(layer_out, layer_one, row)
        print(self.out_weights[0])
        self.__weight_updates(output)
        print(self.out_weights[0])
        # error = None
        # if self.hidden_layers == 0:
        #     error = [row[-1]-values[0][i] for i in range(len(values[0]))]
        # delta = None
        # delta = [self.learn_rate*error[0]*self.weights[0][i] for i in range(len(self.weights[0]))]
        # #update weights
        # self.weights[0] = [self.weights[0][i] + delta[i] for i in range(len(self.weights[0]))]


class Model:
    def __init__(self, train, test, hidden_layers, hidden_nodes, total_classes, learn_rate=0.001):
        self.train = train
        self.test = test
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.learn_rate = learn_rate
        self.total_classes = total_classes
        self.out_weights = []
        self.layer_one_weights = []
        self.layer_two_weights = []
        # Create the network with no hidden layers
        if hidden_layers == 0:
            # Create weights used by the output layer
            for i in range(total_classes):
                temp = [random.uniform(-1.0, 1.0) for i in range(len(train[0]))]
                self.out_weights.append(temp)
        # Create the network with 1 hidden layer
        if hidden_layers == 1:
            # Create the weights for the hidden layer
            for i in range(hidden_nodes):
                temp = [random.uniform(-1.0, 1.0) for i in range(len(train[0]))]
                self.out_weights.append(temp)
            # Create the weights for the output later
            for i in range(total_classes):
                temp = [random.uniform(-1.0, 1.0) for i in range(hidden_nodes)]
                self.layer_one_weights[1].append(temp)
        # Create the network with 2 hidden layers
        if hidden_layers == 2:
            # Create the weights for the 1st hidden layer
            for i in range(hidden_nodes[0]):
                temp = [random.uniform(-1.0, 1.0) for i in range(len(train[0]))]
                self.out_weights.append(temp)
            # Create the weights for the 2nd hidden layer
            for i in range(hidden_nodes[1]):
                temp = [random.uniform(-1.0, 1.0) for i in range(hidden_nodes[0])]
                self.layer_one_weights.append(temp)
            # Create the weights for the output later
            for i in range(total_classes):
                temp = [random.uniform(-1.0, 1.0) for i in range(hidden_nodes[1])]
                self.layer_two_weights.append(temp)

    def __calculate_sigmoid(self, weights, input):
        '''
        This method multplies the weights and input
        and calculates the sigmoid
        '''
        values = [a*b for a, b in zip(weights[1:], input)]
        return utilities.sigmoid(sum(values) + weights[0])

    def derivates(self, value):
        return value*(1-value)
    def __forward_propagate_no_layers(self, row):
        out_layer_values = []
        for label in range(self.total_classes):
            out_layer_values.append(utilities.calculate_sigmoid(self.out_weights[label], row[:-1]))
        return out_layer_values

    def __backpropagate_no_layers(self, out_values, row):
        errors = [row[-1]-out_values[i] for i in range(len(out_values))]
        out_deltas = []
        for label in range(self.total_classes):
            temp = [self.learn_rate*errors[label]*self.out_weights[label][i] for i in range(len(self.out_weights[label]))]
            out_deltas.append(temp)
        return out_deltas

    def __weight_updates(self, updates):
        for label in range(self.total_classes):
            self.out_weights[label] = [self.out_weights[label][i] + updates[label][i] for i in range(len(self.out_weights[label]))]

    def __backpropagate_1_layer(self, out_values, layer_one, expected):
        errors = [expected-out_values[i] for i in range(len(out_values))]
        out_deltas = []
        layer_one = []
        for label in range(self.total_classes):
            temp = [self.learn_rate*errors[label]*self.layer_one_values[i] for i in range(len(self.out_weights[label]))]
            out_deltas.append(temp)

        for node in range(self.hidden_nodes):
            total = 0.0
            for label in range(self.total_classes):
                total += errors[label] * self.out_weights[label][node]
            value = total * self.derivates(self.layer_one_weights[node])
            temp = [value * row[:-1] for i in range(len(row[:-1]))]
        return out_deltas

    def __forward_propagate_1_layers(self, row):
        out_layer_values = []
        layer_one_values = []
        for node in range(self.hidden_nodes):
            out_layer_values.append(utilities.calculate_sigmoid(self.layer_one_weights[node], row[:-1]))

        for label in range(self.total_classes):
            out_layer_values.append(utilities.calculate_sigmoid(self.out_weights[label], layer_one_values))
        return layer_one_values, out_layer_values

    def __forward_propagate_2_layers(self, row):
        out_layer_values = []
        layer_one_values = []
        layer_two_values = []
        for node in range(self.hidden_nodes[0]):
            out_layer_values.append(utilities.calculate_sigmoid(self.layer_one_weights[node], row[:-1]))

        for node in range(self.hidden_nodes[1]):
            out_layer_values.append(utilities.calculate_sigmoid(self.layer_two_weights[node], layer_one_values))

        for label in range(self.total_classes):
            out_layer_values.append(utilities.calculate_sigmoid(self.out_weights[label], layer_two_values))

        return layer_one_values, layer_two_values, out_layer_values

    def __forward_propagate(self, row):
        out_layer_values = None
        layer_one_values = None
        layer_two_values = None
        if self.hidden_layers == 0:
            out_layer_values = self.__forward_propagate_no_layers(row)

        if self.hidden_layers == 1:
            layer_one_values, out_layer_values = self.__forward_propagate_1_layers(row)

        if self.hidden_layers == 2:
            layer_one_values, layer_two_values, out_layer_values = self.__forward_propagate_2_layers(row)

        return layer_one_values, layer_two_values, out_layer_values

    def train_model(self):
        row = self.train[0]
        # Calculate each layer outputs
        layer_one, layer_two, layer_out = self.__forward_propagate(row)
        output = self.__backpropagate_no_layers(layer_out,row[-1])
        print(self.out_weights[0])
        self.__weight_updates(output)
        print(self.out_weights[0])
        # error = None
        # if self.hidden_layers == 0:
        #     error = [row[-1]-values[0][i] for i in range(len(values[0]))]
        # delta = None
        # delta = [self.learn_rate*error[0]*self.weights[0][i] for i in range(len(self.weights[0]))]
        # #update weights
        # self.weights[0] = [self.weights[0][i] + delta[i] for i in range(len(self.weights[0]))]
