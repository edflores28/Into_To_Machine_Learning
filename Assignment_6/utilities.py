import math
import backpropagation_zero
import backpropagation_one
import backpropagation_two


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
    values = []
    for node in range(total_nodes):
        values.append(calculate_sigmoid(weights[node], inputs))
    return values


def network_predict(outputs):
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
        return(outputs.index(max(outputs)))


def create_expected(total_outputs, row):
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
