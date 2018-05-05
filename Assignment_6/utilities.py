import copy
import math
import backpropagation_zero
import backpropagation_one
import backpropagation_two
import rbf_network

'''
The package provides general utilities
'''


def sigmoid(z):
    '''
    This method calculates sigmoid function
    '''
    return (1.0 / (1.0 + math.exp(-z)))


def calculate_derivative(x):
    '''
    This method calculates the derivative of
    the sigmoid function
    '''
    return x * (1 - x)


def distance(x, y):
    '''
    This method calculates the euclidean distance between
    x and y points
    '''
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))


def rbf(x, beta):
    '''
    This method calculates the radial basis function
    '''
    return math.exp(-beta*x)


def calculate_sigmoid(weights, input):
    '''
    This method multplies the weights and input
    and calculates the sigmoid
    '''
    values = [a*b for a, b in zip(weights[1:], input)]
    return sigmoid(sum(values) + weights[0])


def get_train_test_sets(partitions, key):
    '''
    This method creates a training and testing based
    on the partitions key
    '''
    train_list = []
    test_list = []
    for test in partitions:
        if test == key:
            test_list += partitions[key]
        else:
            train_list += partitions[test]
    return train_list, test_list


def transpose(data_list):
    '''
    Transpose the datalist from rows to columns or
    columns to rows
    '''
    return list(map(list, zip(*data_list)))


def calculate_sigmoid_batch(weights, inputs, total_nodes):
    '''
    This method calculates the sigmoid for all nodes
    '''
    values = []
    for node in range(total_nodes):
        values.append(calculate_sigmoid(weights[node], inputs))
    return values


def network_predict(outputs):
    '''
    This method returns a prediction for both
    2 classification and multi classification
    problems
    '''
    # If there is 1 ouput node then its
    # a 2 classification poblem
    if len(outputs) == 1:
        # Return 1 if the prediction is above
        # 0.5
        if outputs[0] > 0.5:
            return 1
        # Otherwise return 0
        else:
            return 0
    # For multi classification problems return
    # the index with the highest value
    else:
        total = sum(outputs)
        softmax = [outputs[i] / (1 + total) for i in range(len(outputs))]
        return(softmax.index(max(softmax)))


def create_expected(total_outputs, row):
    '''
    This method creates a new expected outlist
    and sets the corresponding index of the
    classification
    '''
    expected = [0 for i in range(total_outputs)]
    # For 2 classification problems just set the expected
    # value
    if total_outputs == 1:
        expected[0] = row[-1]
    # Otherwise make sure to set the correct index
    # for the multi classification problem
    else:
        expected[row[-1]] = 1
    return expected


def zero_backprop_test(partitions, outputs):
    '''
    This method iterates through the partitions
    and performs the backpropagation algorithm
    with zero hidden layers
    '''
    error = 0.0
    for key in partitions.keys():
        train, test = get_train_test_sets(partitions, key)
        model = backpropagation_zero.Model(copy.deepcopy(train), copy.deepcopy(test), outputs)
        model.train_model()
        error += model.test_model()
    return error


def one_backprop_test(partitions, hidden_nodes, outputs):
    '''
    This method iterates through the partitions
    and performs the backpropagation algorithm
    with one hidden layers
    '''
    error = 0.0
    for key in partitions.keys():
        train, test = get_train_test_sets(partitions, key)
        model = backpropagation_one.Model(copy.deepcopy(train), copy.deepcopy(test), hidden_nodes, outputs)
        model.train_model()
        error += model.test_model()
    return error


def two_backprop_test(partitions, hidden_nodes, outputs):
    '''
    This method iterates through the partitions
    and performs the backpropagation algorithm
    with two hidden layers
    '''
    error = 0.0
    for key in partitions.keys():
        train, test = get_train_test_sets(partitions, key)
        model = backpropagation_two.Model(copy.deepcopy(train), copy.deepcopy(test), hidden_nodes, outputs)
        model.train_model()
        error += model.test_model()
    return error


def rbf_test(partitions, outputs):
    '''
    This method iterates through the partitions
    and performs the radial basis function network
    '''
    error = 0.0
    for key in partitions.keys():
        train, test = get_train_test_sets(partitions, key)
        model = rbf_network.Model(copy.deepcopy(train), copy.deepcopy(test), outputs)
        model.train_model()
        error += model.test_model()
    return error


def main_test(partitions, h1_nodes, h2_nodes, outputs, rbf_outputs):
    '''
    This method performs the backpropagation testing
    '''
    print("Using the backpropagation with zero hidden layers model")
    input("Press Enter to continue...")
    zero_error = zero_backprop_test(partitions, outputs)
    print("\nTotal accuracy:", zero_error/5, "\n")
    print("Using the backpropagation with one hidden layers model")
    input("Press Enter to continue...")
    one_error = one_backprop_test(partitions, h1_nodes, outputs)
    print("\nTotal accuracy:", one_error/5, "\n")
    print("Using the backpropagation with two hidden layers model")
    input("Press Enter to continue...")
    two_error = two_backprop_test(partitions, h2_nodes, outputs)
    print("\nTotal accuracy:", two_error/5, "\n")
    print("Using the radial basis network model")
    input("Press Enter to continue...")
    rbf_error = rbf_test(partitions, rbf_outputs)
    print("\nTotal accuracy:", rbf_error/5, "\n")
