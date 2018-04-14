import copy
import math
import naive_bayes
import twoclass_lr
import multiclass_lr
import adaline_network

'''
The package provides general utilities
'''


def sigmoid(z):
    '''
    This method calculates sigmoid function
    '''
    return (1.0 / (1.0 + math.exp(-z)))


def signum(x):
    '''
    This method calculates the signum function
    '''
    if x >= 0:
        return 1
    else:
        return 0


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


def naive_test(partitions, labels):
    error = 0.0
    for key in partitions.keys():
        train, test = get_train_test_sets(partitions, key)
        model = naive_bayes.Model(copy.deepcopy(train), copy.deepcopy(test), labels)
        model.train_model()
        # Test the model.
        error += model.test_model()
    return error


def two_class_lr_test(partitions):
    '''
    This method iterates through the partitions
    and performs the two class logistic regression
    algorithm
    '''
    error = 0.0
    for key in partitions.keys():
        train, test = get_train_test_sets(partitions, key)
        model = twoclass_lr.Model(copy.deepcopy(train), copy.deepcopy(test))
        model.train_model()
        error += model.test_model()
    return error


def multi_class_lr_test(partitions, total_classes):
    '''
    This method iterates through the partitions
    and performs the multi class logistic regression
    algorithm
    '''
    error = 0.0
    for key in partitions.keys():
        train, test = get_train_test_sets(partitions, key)
        model = multiclass_lr.Model(copy.deepcopy(train), copy.deepcopy(test), total_classes)
        model.train_model()
        error += model.test_model()
    return error


def adaline_test(partitions, total_classes):
    '''
    This method iterates through the partitions
    and performs the adaline network algorithm
    '''
    error = 0.0
    for key in partitions.keys():
        train, test = get_train_test_sets(partitions, key)
        model = adaline_network.Model(copy.deepcopy(train), copy.deepcopy(test), total_classes)
        model.train_model()
        error += model.test_model()
    return error
