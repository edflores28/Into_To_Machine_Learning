import math

'''
The package provides general utilities
'''

def is_float(value):
    '''
    This method determins whether the value
    is a float
    '''
    try:
        float(value)
        return True
    except ValueError:
        return False


def sigmoid(z):
    '''
    This method calculates sigmoid function
    '''
    return (1.0 / (1.0 + math.exp(-z)))


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
