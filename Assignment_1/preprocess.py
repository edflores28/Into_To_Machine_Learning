import numpy as np
import itertools
import random

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

'''
This swaps the two specified columns, this assumes that
col1 is the id number which is also removed.
'''
def swap_columns(col1, col2, data, rem='True'):
    temp = []
    for entry in data:
        entry[col1], entry[col2] = entry[col2], entry[col1]
        if rem == 'True':
            entry = entry[:-1]
        temp.append(entry)
    return temp

'''
Transpose the datalist from rows to columns or
columns to rows
'''
def transpose(data_list):
    return list(map(list,zip(*data_list)))

'''
This method splits the datalist into 75%
for training data and 25% for testing data1
'''
def split_list(data_list):
    size = int(0.75*len(data_list))
    return data_list[:size], data_list[size:]

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
            column[i] = random.uniform(float(tmin),float(tmax))

    column = [float(i) for i in column]

    bins = np.linspace(float(tmin), float(tmax), TOTAL_BINS)
    x = np.array(column)
    inds = np.digitize(x,bins)
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
    temp = transpose(data)
    for i in range(1, len(temp)):
        temp[i] = digitize(temp[i])
    temp = transpose(temp)
    for  i in range(len(temp)):
        temp[i]= list(itertools.chain.from_iterable(temp[i]))
        temp[i][0] = int(temp[i][0])
    return temp;

'''
This method preprocesses the house data and
generates random values for any missing entries
'''
def convert_house_data(data):
    for i, entry in enumerate(data):
        for j, value in enumerate(entry):
            if value == 'republican' or value == 'n':
                value = 0
            if value == 'democrat' or value == 'y':
                value = 1
            if value == '?':
                value = random.randint(0,1)
            data[i][j] = value;

'''
This method converts the class/result column into discrete
values
'''
def convert_column(data_list, one, zero):
    for i in range(len(data_list)):
        if data_list[i][0] == one:
            data_list[i][0] = 1
        if data_list[i][0] == zero:
            data_list[i][0] = 0

'''
This method creates updated row entries for table_list
with 3+ classes
'''
def build_multiclass(data, data_dict):
    temp = []
    for i in range(len(data)):
        data[i][0] = data_dict[data[i][0]]
