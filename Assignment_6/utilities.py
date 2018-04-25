import math

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
    print(sum(values) + weights[0])
    try:
        return sigmoid(sum(values) + weights[0])
    except:
        # print(input)
        # print(weights,"\n")
        # print(values)
        # print(weights[0])
        # print(sum(values))
        return 1.0


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


def outer_layer_backprop(errors, layer, rate, total_nodes):
    deltas = []
    for node in range(total_nodes):
        deltas.append([rate*errors[node]*layer[i] for i in range(len(layer))])
    return deltas

# def hidden_layer_backprop(errors, weights_a, weights_b, layer, rate, total_nodes):
#     deltas = []
#     for node in range(total_nodes):
#         total = 0.0
#         for i in range(len(errors)):
#             total += errors[i] * weights_a[i][node]
#         value = [total * calculate_derivative(weights_b[node][i]) for i in range(len(weights_b[node]))]
#         temp_row = [1.0] + row[:-1]
#         temp = [value[i] * temp_row[i] for i in range(len(temp_row))]
#         deltas.append(temp)
