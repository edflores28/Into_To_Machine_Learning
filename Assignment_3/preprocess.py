import re
import copy
import random

'''
Transpose the datalist from rows to columns or
columns to rows
'''
def transpose(data_list):
    return list(map(list,zip(*data_list)))

'''
This method reads a file line by line and discards
the /n character
'''
def read_file(filename,split=None):
    temp = []
    with open(filename) as file:
        for line in file:
            if split != None:
                temp.append(line.strip().split(split))
            else:
                line = re.sub(' +',' ',line.strip())
                temp.append(line.split(" "))
    return temp

def create_partitions(total_parts, class_index, data):
    class_dict = {}
    column_length = len(data)
    partition_length = int(round(column_length/total_parts,0))
    # Swap to column representation
    data = transpose(data)
    # Create keys for the dictionary
    for entry in data[class_index]:
        class_dict[entry] = []
    class_sizes = copy.deepcopy(class_dict)
    # Swap back to row representation
    data = transpose(data)
    # Iterate through the data set and add the entries
    # to the class dictionary
    for row in range(len(data)):
        class_dict[data[row][class_index]].append(row)
    # Calculte the percentages for each class and shuffle thhe lists
    for key in class_dict:
        class_sizes[key] = len(class_dict[key])/column_length
        random.shuffle(class_dict[key])
    part_dict = {}
    for partition in range(total_parts):
        
    print(class_sizes.values(), partition_length, column_length)
