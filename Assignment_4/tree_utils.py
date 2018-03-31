import random

'''
This package provides utilities for decision trees
'''


def print_node(node):
    '''
    This method iterates through the node and
    prints out information.
    '''
    if node is None:
        return
    if node.get_leaf():
        print("Leaf value:", node.get_label())
        return
    else:
        print("Decision Node:", node.get_feature_index())
        branches = node.get_branches()
        for key in branches:
            print(key)
            print_node(branches[key])


def determine_label(test, root):
    '''
    This method takes in the root of the tree and
    a test entry and traverses the tree until a
    label is found
    '''
    # If the root root is a leaf root return the label
    if root.get_leaf() is True:
        return root.get_label()
    # Otherwise continue iterating through the tree
    else:
        index = root.get_feature_index()
        value = test[index]
        branches = root.get_branches()
        threshold = root.get_threshold()
        # If the branches are catergorical then use the value
        # as the key for branches
        if threshold is None:
            # On misclassification return 0.0
            try:
                return determine_label(test, branches[value])
            except:
                return 0.0
        # Otherwise use the threshold
        else:
            # When the value is less than or equal to the threshold
            # use the left branch for recursion
            if value <= threshold:
                try:
                    return determine_label(test, branches['left'])
                except:
                    return 0.0
            # Otherwise use the right branch for recursion
            else:
                # On misclassification return 0.0
                try:
                    return determine_label(test, branches['right'])
                except:
                    return 0.0


def predict_accuracy(test, root):
    '''
    This method takes the testing set and the
    root node of the tree. Each entry is validated
    based on the prediction and the % correct is
    returned
    '''
    correct = 0
    incorrect = 0
    for entry in range(len(test)):
        x = determine_label(test[entry], root)
        if x == test[entry][-1]:
            correct += 1
        else:
            incorrect += 1
    return(correct*100)/(incorrect+correct)


def get_pruning_set(train):
    '''
    This method removes 10% of the training set in order
    to create a pruning set
    '''
    prune_length = int(round(len(train)*.10, 0))
    prune_set = []
    # Pop random index values from the training set
    # and add them to the pruning set.
    while len(prune_set) < prune_length:
        index = random.sample(range(len(train)), 1)
        prune_set.append(train.pop(index[0]))
    return prune_set, train
