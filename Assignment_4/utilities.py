import math


def entropy(p):
    return -p*math.log2(p)


def distance(x, y):
    '''
    This method calculates the euclidean distance between c_points
    '''
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))


def transpose(data_list):
    '''
    Transpose the datalist from rows to columns or
    columns to rows
    '''
    return list(map(list, zip(*data_list)))
