import math
'''
This method calculates the euclidean distance between c_points
'''
def distance (x, y):
        return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))

'''
Transpose the datalist from rows to columns or
columns to rows
'''
def transpose(data_list):
    return list(map(list,zip(*data_list)))
