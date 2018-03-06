import utilities
import random
def cNN(data, train_set, class_index):
    new_train = []
    init = train_set.pop(random.randint(0,len(train_set) - 1))
    new_train.append(init)
    count = 0;
    while count != len(train_set):
        exit = True
        prev_length = len(new_train)
        index = random.randint(0, len(train_set) - 1)
        label = kNN(data, new_train, index, class_index, 1)
        if label != data[index][class_index]:
            remove = train_set.pop(index)
            new_train.append(remove)
            count = 0
        else:
            count += 1

def kNN(data, train_set, point, class_index, k):
    distances = []
    neighbors = []
    # Calculate all the distances
    for entry in train_set:
        dist = utilities.distance(data[point][:class_index],data[entry][:class_index])
        distances.append(dist)
    # Find k closest distances and create the neighbors list
    for i in range(k):
        value = distances.index(min(distances))
        del distances[value]
        neighbors.append(value)
    votes_dict = {}
    # Count the neighbors votes
    for entry in neighbors:
        if data[train_set[entry]][class_index] not in votes_dict:
            votes_dict[data[train_set[entry]][class_index]] = 0
        votes_dict[data[train_set[entry]][class_index]] += 1
    return max(votes_dict, key=votes_dict.get)
