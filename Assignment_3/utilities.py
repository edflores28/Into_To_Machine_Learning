import math
import nearest_neighbor

'''
This method calculates the euclidean distance between
x and y points
'''
def distance (x, y):
        return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))

'''
Transpose the datalist from rows to columns or
columns to rows
'''
def transpose(data_list):
    return list(map(list,zip(*data_list)))

'''
This method normalizes te values in each feature where:
new value = (old value - feature min)/(feature max - feature min)
'''
def normalize_data(data):
    data = transpose(data)
    for feature in range(len(data[:-1])):
        diff = max(data[feature]) - min(data[feature])
        data[feature] = [(data[feature][value]-min(data[feature]))/diff if diff > 0 else 0.0 for value in range(len(data[feature]))]
    return transpose(data)
'''
This method creates a training and testing based
on the partitions key
'''
def get_train_test_sets(partitions,key):
    part = {}
    train_list = []
    test_list = []
    for test in partitions:
        if test == key:
            test_list += partitions[key]
        else:
            train_list += partitions[test]
    return train_list, test_list

'''
This method attempts to find the optimal k nearest neighbor
for the data set
'''
def optimal_k(data,class_index,partition_dict,classify=True,cNN=False):
    k_list = [i for i in range(1,35)]
    accuracies = []
    for k in k_list:
        accuracy = 0.0
        for key in partition_dict:
            train, test = get_train_test_sets(partition_dict,key)
            if cNN:
                train = nearest_neighbor.cNN(data, train, len(data[0])-1)
            accuracy += nearest_neighbor.kNN_Test(data, train, test,class_index,k,classify)
        accuracies.append(accuracy/5.0)
        accuracy = 0.0
    if classify:
        print(max(accuracies), k_list[accuracies.index(max(accuracies))])
    else:
        print(min(accuracies), k_list[accuracies.index(min(accuracies))])
    print(accuracies)
