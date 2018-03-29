import preprocess
import utilities
import id3


def print_node(root):
    if root is None:
        return
    if root.get_leaf():
        print("Leaf value:", root.get_label())
        return
    else:
        print("Decision Node:", root.get_feature_index())
        branches = root.get_branches()
        for key in branches:
            print(key)
            print_node(branches[key])


def predict(test, root):
    # If the root node is a leaf node return the label
    if root.get_leaf() is True:
        value = root.get_label()
        return root.get_label()
    else:
        index = root.get_feature_index()
        value = test[index]
        branches = root.get_branches()
        threshold = root.get_threshold()
        if threshold is None:
            try:
                return predict(test, branches[value])
            except:
                print("except")
                return 0.0
        else:
            if value < threshold:
                return predict(test, branches['left'])
            else:
                return predict(test, branches['right'])


# Obtain the dataset
dataset = preprocess.read_file(filename="./abalone.data", split=',')
# Obtain a column representation
dataset = utilities.transpose(dataset)
# For continous data convert them from strings to float
for entry in range(len(dataset)):
    if utilities.is_float(dataset[entry][0]):
        dataset[entry] = [float(s.replace(',', '')) for s in dataset[entry]]
# Obtain the row representation
dataset = utilities.transpose(dataset)
train = dataset[:3100]
test = dataset[3100:]
feature_indices = [i for i in range(len(dataset[0]))]
x = id3.ID3(train, feature_indices)
y = x.build_tree()

correct = 0
incorrect = 0
for entry in range(len(test)):
    x = predict(test[entry],y)
    if x == test[entry][-1]:
        correct += 1
    else:
        incorrect +=1
print((correct*100)/(incorrect+correct))
