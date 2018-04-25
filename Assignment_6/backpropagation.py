import random
import utilities


class ZeroLayer:
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

    def __forward_propagate(self, row):
        out_layer_values = []
        for label in range(self.total_classes):
            out_layer_values.append(utilities.calculate_sigmoid(self.out_weights[label], row[:-1]))
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
            self.out_weights[label] = [self.out_weights[label][i] + updates[label][i] for i in range(len(self.out_weights[label]))]

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
        out_deltas = utilities.outer_layer_backprop(errors, layer_one_values, self.learn_rate, self.total_classes)
        layer_one = []

        for node in range(self.hidden_nodes):
            total = 0.0
            for label in range(self.total_classes):
                total += errors[label] * self.out_weights[label][node]
            value = [total * utilities.calculate_derivative(self.layer_one_weights[node][i]) for i in range(len(self.layer_one_weights[node]))]
            temp_row = [1.0] + row[:-1]
            temp = [value[i] * temp_row[i] for i in range(len(temp_row))]
            layer_one.append(temp)
        return out_deltas, layer_one

    def __forward_propagate(self, row, test=False):
        layer_one_values = utilities.calculate_sigmoid_batch(self.layer_one_weights, row[:-1], self.hidden_nodes)
        out_layer_values = utilities.calculate_sigmoid_batch(self.out_weights, layer_one_values, self.total_classes)
        if test:
            return out_layer_values
        return layer_one_values, out_layer_values

    def __weight_updates(self, output, layer):
        for label in range(self.total_classes):
            temp = [self.out_weights[label][i] + output[label][i] for i in range(len(self.out_weights[label]))]
            self.out_weights[label] = temp
        for node in range(self.hidden_nodes):
            temp = [self.layer_one_weights[node][i] + layer[node][i] for i in range(len(self.layer_one_weights[node]))]
            self.layer_one_weights[node] = temp

    def train_model(self):
        row = self.train[0]
        # for row in self.train:
        # Calculate each layer outputs
        one_prop, out_prop = self.__forward_propagate(row)
        print(one_prop, out_prop)
        out_back, one_back = self.__backpropagate(out_prop, one_prop, row)
        print(out_back, one_back)
        self.__weight_updates(out_back, one_back)

    # def test_model(self):
    #     for row in self.test:
    #         output = self.__forward_propagate(row, True)
    #         #print(output, row[-1])

class TwoLayer:
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

    def __forward_propagate(self, row):
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
