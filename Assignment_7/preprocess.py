'''
This package contrains methods that are used to preprocess
the data set
'''


def read_file(filename):
    '''
    This method reads a file line by line and discards
    the /n character
    '''
    temp = []
    with open(filename) as file:
        for line in file:
            temp.append(line.strip().split(','))
    return temp
