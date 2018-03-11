import utilities
import random
import math

MAX_PRINT = 1
MAX_LEN = 4
COUNTER = 0

'''
This method creates a new training set using condensed nearest
neighbor.
'''
def cNN(data, train_set, class_index):
    global MAX_PRINT
    count = 0
    print("Starting condensed nearest neighbor")
    new_train = []
    # Add a random datapoint in the training set
    init = train_set.pop(random.randint(0,len(train_set) - 1))
    new_train.append(init)
    exit = 0;
    # When count reaches 60 there has be no updates to the
    # new training list so break the loop.
    while exit != 60:
        # Obtain a random point in the training set
        # and obtainin the nearest neighbor
        index = random.randint(0, len(train_set) - 1)
        if count < MAX_PRINT:
            print("Selected point at index ", index)
        label = kNN(data, new_train, index, class_index, 1)
        # Check to see if the classification is different
        # between the nearest neighbor and the data point
        # if it is add the data point to the new list
        if label != data[index][class_index]:
            remove = train_set[index]
            new_train.append(remove)
            del train_set[index]
            exit = 0
        if count < MAX_PRINT:
            print("Added index", index,"to new list")
        else:
            exit += 1
        count += 1
    print("Finished condensed nearest neighbor")
    return new_train

'''
This method computes the distances from the data points
to all entries in the training set
'''
def get_distances(data,train_set,point,class_index,kmeans=False):
    global COUNTER
    global MAX_PRINT
    global MAX_LEN
    distances = []
    # Calculate all the distances
    for entry in train_set:
        if kmeans:
            dist = utilities.distance(data[point][:class_index],entry[:class_index])
        else:
            dist = utilities.distance(data[point][:class_index],data[entry][:class_index])
        distances.append(dist)
    if COUNTER < MAX_PRINT:
        print("Distances calculated",distances[:MAX_LEN])
    return distances

'''
This method performs k nearest neighbor
'''
def kNN(data, train_set, point, class_index, k, classify=True,kmeans=False):
    global MAX_PRINT
    global COUNTER
    neighbors = {}
    distances = get_distances(data,train_set,point,class_index,kmeans)
    # Find k closest distances and create the neighbors list
    for i in range(k):
        neighbors[distances.index(min(distances))] = min(distances)
        del distances[distances.index(min(distances))]
    # Perform the following processing for classification
    if classify:
        votes_dict = {}
        # Calculate 1/distance and add them to it's class to obtain the
        # weighed sum.
        for entry in neighbors:
            # If kmeans is turned on look at the clusters for
            # their class information
            if kmeans:
                if train_set[entry][class_index] not in votes_dict:
                    votes_dict[train_set[entry][class_index]] = 0
                # Protect agains division by 0
                if neighbors[entry] != 0.0:
                    votes_dict[train_set[entry][class_index]] += (1/neighbors[entry])
                else:
                    votes_dict[train_set[entry][class_index]] += 0.0
            # Otherwise look at the data set for class information
            else:
                if data[train_set[entry]][class_index] not in votes_dict:
                    votes_dict[data[train_set[entry]][class_index]] = 0
                if neighbors[entry] != 0.0:
                    votes_dict[data[train_set[entry]][class_index]] += (1/neighbors[entry])
                else:
                    votes_dict[data[train_set[entry]][class_index]] += 0.0
        # Iterate through the votes and find the class
        # that has the highest value
        if COUNTER < MAX_PRINT:
            print("Final vote count between classes")
            print(votes_dict)

        max_value = -math.inf
        max_class = 0
        for key in votes_dict:
            if votes_dict[key] > max_value:
                max_class= key
                max_value = votes_dict[key]
        if COUNTER < MAX_PRINT:
            print("Class with highest votes is", max_class)
        return max_class
    # Perform the following for regression. The prediction is based
    # on the average value of the nearest neighbors
    else:
        predict = 0
        # Iterate through the neighbors and compute the average for
        # the regression value.
        for entry in neighbors:
            if kmeans:
                predict += train_set[entry][class_index]
            else:
                predict += data[train_set[entry]][class_index]
        predict /= k
        if COUNTER < MAX_PRINT:
            print("Average class value between the neighbors", predict)
        return predict

'''
This method predicts the class for the test_set provides
the accuracy of correct predictions for classification
or the mean square error for regression problems
'''
def kNN_Test(data, train_set, test_set, class_index, k,classify=True,kmeans=False):
    global MAX_PRINT
    global COUNTER
    correct = 0
    incorrect = 0
    squares = 0.0
    # Iterate through the test set and predict their values
    for entry in range(len(test_set)):
        if COUNTER < MAX_PRINT:
            print("Starting k nearest neighbor")
        predict = kNN(data,train_set,test_set[entry],class_index,k,classify,kmeans)
        if COUNTER < MAX_PRINT:
            print("Finshed k nearest_neighbor")
        # If we are doing classification the count the number of
        # correct and incorrect predictions
        if classify:
            if COUNTER < MAX_PRINT:
                print("Predicted value:", predict, "Expected value:",data[test_set[entry]][class_index])
            if predict == data[test_set[entry]][class_index]:
                correct += 1
                if COUNTER < MAX_PRINT:
                    print("Made a correct guess")
            else:
                incorrect += 1
                if COUNTER < MAX_PRINT:
                    print("Made an incorrect guess")
        # If we are doing regression add the determine the squared
        # error.
        else:
            if COUNTER < MAX_PRINT:
                print("Predicted value:", predict, "Expected value:",data[test_set[entry]][class_index])
            squares += (data[test_set[entry]][class_index]-predict)**2
        COUNTER += 1
    # Return the % correct for classification and the mean square error
    # for regression
    COUNTER = 0
    if classify:
        return(correct/(incorrect+correct))*100.0
    else:
        return(squares/len(test_set))
