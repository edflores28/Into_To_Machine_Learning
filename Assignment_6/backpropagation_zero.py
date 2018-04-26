import random
import utilities
import copy

class Model:
    def __init__(self, train, test, outputs, learn_rate=0.001):
        self.train = train
        self.test = test
        self.learn_rate = learn_rate
        self.outputs = outputs
        self.weightsO = []
        self.weightsI = []
        self.layer_two_weights = []
        # Create weights used by the output layer
        for i in range(outputs):
            temp = [random.uniform(-1.0, 1.0) for i in range(len(train[0]))]
            self.weightsO.append(temp)

    def __forward_propagate(self, row):
        out_layer_values = []
        for label in range(self.outputs):
            out_layer_values.append(utilities.calculate_sigmoid(self.weightsO[label], row[:-1]))
        return out_layer_values

    def __backpropagate(self, out_values, row):
        errors = [row[-1]-out_values[i] for i in range(len(out_values))]
        out_deltas = []
        temp_row = [1.0] + row[:-1]
        for label in range(self.outputs):
            temp = [self.learn_rate*errors[label]*temp_row[i] for i in range(len(temp_row))]
            out_deltas.append(temp)
        return out_deltas

    def __weight_updates(self, updates):
        for label in range(self.outputs):
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
