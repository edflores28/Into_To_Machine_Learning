import re
import copy
import utilities


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


def create_partitions(total_parts, class_index, data):
    '''
    This method creates a partition dictionary based
    on the dataset
    '''
    class_dict = {}
    column_length = len(data)
    partition_length = int(round(column_length/total_parts, 0))
    # Swap to column representation
    data = utilities.transpose(data)
    # Create keys for the dictionary
    for entry in data[class_index]:
        class_dict[entry] = []
    class_sizes = copy.deepcopy(class_dict)
    # Swap back to row representation
    data = utilities.transpose(data)
    # Iterate through the data set and add the entries
    # to the class dictionary
    for row in range(len(data)):
        class_dict[data[row][class_index]].append(row)
    # Calculte the percentages for each class and shuffle thhe lists
    for key in class_dict:
        temp_pct = len(class_dict[key])/column_length
        class_sizes[key] = round(partition_length * temp_pct, 0)
        # random.shuffle(class_dict[key])
    part_dict = {}
    # Build the partitions
    for partition in range(total_parts):
        part_dict[partition] = []
        for key in class_dict:
            part_dict[partition] += build_partition(class_dict[key], class_sizes[key])
    return part_dict
