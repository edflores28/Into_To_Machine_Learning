import re

'''
This package provides utilities for data set preprocessing
'''


def swap_columns(col1, col2, data, rem='True'):
    '''
    This swaps the two specified columns, this assumes that
    col1 is the id number which is also removed.
    '''
    temp = []
    for entry in data:
        entry[col1], entry[col2] = entry[col2], entry[col1]
        if rem == 'True':
            entry = entry[:-1]
        temp.append(entry)
    return temp


def read_file(filename, split=None, rem_list=None, class_idx=0):
    '''
    This method reads a file line by line and discards
    the /n character
    '''
    temp = []
    with open(filename) as file:
        for line in file:
            if split is not None:
                temp_line = line.strip().split(split)
            else:
                line = re.sub(' +', ' ', line.strip())
                temp_line = line.split(" ")
            if rem_list is None:
                temp.append(temp_line)
            else:
                if not temp_line[class_idx] in rem_list:
                    temp.append(temp_line)
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
