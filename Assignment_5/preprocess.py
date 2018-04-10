import numpy as np
import itertools
import random
import utilities


TOTAL_BINS = 4

'''
This method reads a file line by line and discards
the /n character
'''
def read_file(filename):
    temp = []
    with open(filename) as file:
        for line in file:
            temp.append(line.strip().split(','))
    return temp


def swap_columns(col1, col2, data, rem=True):
    '''
    This swaps the two specified columns, this assumes that
    col1 is the id number which is also removed.
    '''
    temp = []
    for entry in data:
        entry[col1], entry[col2] = entry[col2], entry[col1]
        if rem:
            entry = entry[:-1]
        temp.append(entry)
    return temp


def build_partition(class_list, class_size):
    '''
    This method takes the class list and it's size
    based on the partition length and creates abs
    list of values
    '''
    temp = []
    pop_counter = 0
    while pop_counter < class_size and class_list:
        val = class_list.pop(0)
        temp.append(val)
        pop_counter += 1
    return temp


def create_partitions(total_parts, data):
    '''
    This method patitions the dataset into total_parts
    and returns a dictionary
    '''
    column_length = len(data)
    partition_length = int(round(column_length/total_parts, 0))
    part_dict = {}
    for i in range(total_parts):
        part_dict[i] = build_partition(data, partition_length)
    return part_dict


def normalize_data(column):
    minimum = min(column)
    maximum = max(column)
    for entry in range(len(column)):
        try:
            column[entry] = (column[entry]-minimum)/(maximum-minimum)
        except ZeroDivisionError:
            column[entry] = 0.0

def get_minmax(column):
    if '?' in column:
        missing = column.index('?')
        return min(column[:missing]), max(column[:missing])
    return min(column), max(column)

'''
This method takes a column converts it to float
and creates a discrete list based on TOTAL_BINS
it also generates a random number for the dataset
if a ? is encountered
'''
def digitize(column):
    tmin, tmax = get_minmax(column)
    for i, entry in enumerate(column):
        if entry == '?':
            column[i] = random.uniform(float(tmin), float(tmax))
    column = [float(i) for i in column]
    bins = np.linspace(float(tmin), float(tmax), TOTAL_BINS)
    x = np.array(column)
    inds = np.digitize(x, bins)
    disc_list = []
    for i in range(len(column)):
        disc_list.append([])
        disc_list[i] = [0 for i in range(TOTAL_BINS)]
        disc_list[i][inds[i]-1] = 1
    return disc_list

'''
This method takes a data list and coverts the whole dataset
minus the class column into a discrete test_list
'''
def build_discrete(data):
    temp = utilities.transpose(data)
    for i in range(len(temp)):
        temp[i] = digitize(temp[i])
    temp = utilities.transpose(temp)
    for i in range(len(temp)):
        temp[i] = list(itertools.chain.from_iterable(temp[i]))
        temp[i][0] = int(temp[i][0])
    return temp


def convert_house_data(data):
    '''
    This method preprocesses the house data and
    generates random values for any missing entries
    '''
    for i, entry in enumerate(data):
        for j, value in enumerate(entry):
            if value == 'republican' or value == 'n':
                value = 0
            if value == 'democrat' or value == 'y':
                value = 1
            if value == '?':
                value = random.randint(0,1)
            data[i][j] = value


def convert_column(data_list, one, zero):
    '''
    This method converts the class/result column into discrete
    values, where one is the value to convert to 1 and zero is
    the value to convert to zero.
    '''
    for i in range(len(data_list)):
        if data_list[i][-1] == one:
            data_list[i][-1] = 1
        if data_list[i][-1] == zero:
            data_list[i][-1] = 0


def fill_column(column):
    if '?' in column:
        temp = [int(x) for x in column if x != '?']
        minimum = min(temp)
        maximum = max(temp)
        for entry in range(len(column)):
            if column[entry] == '?':
                column[entry] = random.uniform(minimum, maximum)

'''
This method creates updated row entries for table_list
with 3+ classes
'''
def build_multiclass(data, data_dict):
    temp = []
    for i in range(len(data)):
        data[i][-1] = data_dict[data[i][-1]]
