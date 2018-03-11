import re
import copy
import random
import utilities

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
This method reads a file line by line and discards
the /n character
'''
def read_file(filename,split=None,rem_list=None,class_idx=0):
    temp = []
    with open(filename) as file:
        for line in file:
            if split != None:
                temp_line = line.strip().split(split)
            else:
                line = re.sub(' +',' ',line.strip())
                temp_line = line.split(" ")
            if rem_list == None:
                temp.append(temp_line)
            else:
                if not temp_line[class_idx] in rem_list:
                    temp.append(temp_line)
    return temp

'''
This method takes the class list and it's size
based on the partition length and creates abs
list of values
'''
def build_partition(class_list, class_size):
    temp = []
    pop_counter = 0
    while pop_counter < class_size and class_list:
        val = class_list.pop(0)
        temp.append(val)
        pop_counter += 1
    return temp

'''
This method creates a partition dictionary based
on the dataset
'''
def create_partitions(total_parts, class_index, data):
    class_dict = {}
    column_length = len(data)
    partition_length = int(round(column_length/total_parts,0))
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
        class_sizes[key] = round(partition_length * temp_pct,0)
        #random.shuffle(class_dict[key])
    part_dict = {}
    # Build the partitions
    for partition in range(total_parts):
        part_dict[partition] = []
        for key in class_dict:
            part_dict[partition] += build_partition(class_dict[key],class_sizes[key])
    return part_dict

def create_regress_partitions(total_parts, data):
    column_length = len(data)
    partition_length = int(round(column_length/total_parts,0))
    index_list = [i for i in range(len(data))]
    part_dict = {}
    random.shuffle(index_list)
    for i in range(total_parts):
        part_dict[i] = build_partition(index_list,partition_length)
    return part_dict
