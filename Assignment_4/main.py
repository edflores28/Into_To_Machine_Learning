import preprocess
import utilities
import id3
import prune
import tree_utils
import copy


def do_abalone():
    total_parts = 5
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
    # Obtain the total features
    features = len(dataset[0])
    # Partition the data for cross validation
    parts = preprocess.create_partitions(total_parts, dataset)
    # Generate the feature indices and accuracies
    feature_indices = [i for i in range(features)]
    accuracy = 0.0
    p_accuracy = 0.0
    # Iterate through each partition and test
    for key in parts:
        # Obtain a training and testing set
        train, test = utilities.get_train_test_sets(parts, key)
        # Copy the training set
        c_train = copy.deepcopy(train)
        # Obtain a pruning set from the training set
        prune_set, c_train = tree_utils.get_pruning_set(c_train)
        # Copy the training set
        c_test = copy.deepcopy(test)
        # Copy the features
        c_features = copy.deepcopy(feature_indices)
        # Perform the ID3 algorithm on the training set
        tree = id3.ID3(c_train, c_features)
        root = tree.build_tree()
        # Obtain the accuracy
        accuracy += tree_utils.predict_accuracy(c_test, root)
        # Prun the created tree with the pruning set
        prune_tree = prune.Prune(root, prune_set)
        # Obtain the accuracy
        p_accuracy += prune_tree.reduced_error_prune()
    print("Percentage of total correct without pruning:", accuracy/total_parts)
    print("Percentage of total correct with pruning:", p_accuracy/total_parts)


def do_segmentation():
    total_parts = 5
    dataset = preprocess.read_file(filename="segmentation.data", split=',')
    # Remove the first 5 entries since they are garbage
    dataset = dataset[5:]
    # Swap the columns
    dataset = preprocess.swap_columns(0, len(dataset[0])-1, dataset, False)
    # Obtain a column representation
    dataset = utilities.transpose(dataset)
    # For continous data convert them from strings to float
    for entry in range(len(dataset)):
        if utilities.is_float(dataset[entry][0]):
            dataset[entry] = [float(s.replace(',', '')) for s in dataset[entry]]
    # Obtain the row representation
    dataset = utilities.transpose(dataset)
    for i in range (5):
        print(dataset[i])
    # Obtain the total features
    features = len(dataset[0])
    # Partition the data for cross validation
    parts = preprocess.create_partitions(total_parts, dataset)
    # Generate the feature indices and accuracies
    feature_indices = [i for i in range(features)]
    accuracy = 0.0
    p_accuracy = 0.0
    # Iterate through each partition and test
    for key in parts:
        print(key)
        # Obtain a training and testing set
        train, test = utilities.get_train_test_sets(parts, key)
        # Copy the training set
        c_train = copy.deepcopy(train)
        # Obtain a pruning set from the training set
        prune_set, c_train = tree_utils.get_pruning_set(c_train)
        # Copy the training set
        c_test = copy.deepcopy(test)
        # Copy the features
        c_features = copy.deepcopy(feature_indices)
        # Perform the ID3 algorithm on the training set
        tree = id3.ID3(c_train, c_features)
        root = tree.build_tree()
        # Obtain the accuracy
        accuracy += tree_utils.predict_accuracy(c_test, root)
        # Prun the created tree with the pruning set
        prune_tree = prune.Prune(root, prune_set)
        # Obtain the accuracy
        p_accuracy += prune_tree.reduced_error_prune()
    print("Percentage of total correct without pruning:", accuracy/total_parts)
    print("Percentage of total correct with pruning:", p_accuracy/total_parts)


def do_car():
    total_parts = 5
    # Obtain the dataset
    dataset = preprocess.read_file(filename="./car.data", split=',')
    # Obtain the total features
    features = len(dataset[0])
    # Partition the data for cross validation
    parts = preprocess.create_partitions(total_parts, dataset)
    # Generate the feature indices and accuracies
    feature_indices = [i for i in range(features)]
    accuracy = 0.0
    p_accuracy = 0.0
    # Iterate through each partition and test
    for key in parts:
        # Obtain a training and testing set
        train, test = utilities.get_train_test_sets(parts, key)
        # Copy the training set
        c_train = copy.deepcopy(train)
        # Obtain a pruning set from the training set
        prune_set, c_train = tree_utils.get_pruning_set(c_train)
        # Copy the training set
        c_test = copy.deepcopy(test)
        # Copy the features
        c_features = copy.deepcopy(feature_indices)
        # Perform the ID3 algorithm on the training set
        tree = id3.ID3(c_train, c_features)
        root = tree.build_tree()
        # Obtain the accuracy
        accuracy += tree_utils.predict_accuracy(c_test, root)
        # Prun the created tree with the pruning set
        prune_tree = prune.Prune(root, prune_set)
        # Obtain the accuracy
        p_accuracy += prune_tree.reduced_error_prune()
    print("Percentage of total correct without pruning:", accuracy/total_parts)
    print("Percentage of total correct with pruning:", p_accuracy/total_parts)

do_segmentation()
#do_abalone()
#do_car()
