def print_node(node):
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


def determine_label(test, node):
    # If the node node is a leaf node return the label
    if node.get_leaf() is True:
        return node.get_label()
    else:
        index = node.get_feature_index()
        value = test[index]
        branches = node.get_branches()
        threshold = node.get_threshold()
        if threshold is None:
            try:
                return determine_label(test, branches[value])
            except:
                print("except")
                return 0.0
        else:
            if value < threshold:
                return determine_label(test, branches['left'])
            else:
                return determine_label(test, branches['right'])


def predict_accuracy(test, node):
    correct = 0
    incorrect = 0
    for entry in range(len(test)):
        x = determine_label(test[entry], node)
        if x == test[entry][-1]:
            correct += 1
        else:
            incorrect +=1
    return(correct*100)/(incorrect+correct)
