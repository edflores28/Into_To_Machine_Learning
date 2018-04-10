import copy
import preprocess
import multiclass_lr
import twoclass_lr
import utilities
import random
import naive_bayes


TOTAL_PARTITIONS = 5
def do_house_votes():
    '''
    This method executes the house votes dataset test.
    '''
    total_parts = 5
    print("The house votes dataset will be processed and tested.")
    # Read the data for the specified file
    dataset = preprocess.read_file("./house-votes-84.data")
    # Convert the house data
    preprocess.convert_house_data(dataset)
    # Swap the class column with the last column
    preprocess.swap_columns(0, len(dataset[0])-1, dataset, False)
    # Randomize the dataset
    #random.shuffle(dataset)
    # Set the default house_weights
    #winnow_weights = winnow.set_default_weights(dataset[0][1:])
    # Split the list into a training and test list
    #train, test = preprocess.split_list(dataset)
    # Train for winnow
    #winnow_weights = winnow.train(winnow_weights,train)
    # Test the model out.
    #winnow.test(winnow_weights,test)
    # Train for naive bayes
    #naive_table = naive_bayes.build_table(train)
    # Test the model out
    #naive_bayes.test(test, naive_table)
    #dataset = dataset[:5]
    print("Creating a total of", total_parts, "partitions")
    parts = preprocess.create_partitions(TOTAL_PARTITIONS, dataset)
    for key in parts.keys():
        train, test = utilities.get_train_test_sets(parts, key)
        log_reg = twoclass_lr.LG(copy.deepcopy(train), copy.deepcopy(test))
        log_reg.train_model()
        log_reg.test_model()


def do_iris():
    print("The iris dataset will be processed and tested.")
    # Read the data for the specified file
    dataset = preprocess.read_file("./iris.data")
    # Swap the specified columns in order to have the classifications
    # in the first column.
    #dataset = preprocess.swap_columns(0,4,dataset,False)
    dataset = utilities.transpose(dataset)
    # For continous data convert them from strings to float
    for entry in range(len(dataset)):
        if utilities.is_float(dataset[entry][0]):
            dataset[entry] = [float(s.replace(',', '')) for s in dataset[entry]]
            preprocess.normalize_data(dataset[entry])

    dataset = utilities.transpose(dataset)

    iris_dict = {'Iris-virginica': 0,
                 'Iris-versicolor': 1,
                 'Iris-setosa': 2}
    preprocess.build_multiclass(dataset, iris_dict)
    #random.shuffle(dataset)


    parts = preprocess.create_partitions(TOTAL_PARTITIONS, dataset)
    for key in parts.keys():
        train, test = utilities.get_train_test_sets(parts, key)
        test = multiclass_lr.LR(train, test, 3)
        test.train_model()
        test.test_model()
    # Convert the real data present in the dataset and diiscretize them.
    # dataset = preprocess.build_discrete(dataset)
    #
    #
    # # Build the naive bayes likleihood table.
    # naive_table = naive_bayes.build_table(train)
    # # Test the model.
    # naive_bayes.test(test, naive_table)


def do_breast_cancer():
    '''
    This method executes the breast cancer dataset test.
    '''
    print("The breast dataset will be processed and tested.")
    # Read the data for the specified file
    dataset = preprocess.read_file("./breast-cancer-wisconsin.data")
    # Obtain column representation
    dataset = utilities.transpose(dataset)
    # Iterate through each column and fill in any missing entries
    # and convert the string to integers
    # for entry in range(len(dataset)):
    #     preprocess.fill_column(dataset[entry])
    #     dataset[entry] = [int(i) for i in dataset[entry]]
    #     if entry != len(dataset) - 1:
    #         preprocess.normalize_data(dataset[entry])
    # Delete index 0 since this is the ID field and is not needed
    del dataset[0]
    # Obtain row representation
    dataset = utilities.transpose(dataset)
    dataset = preprocess.build_discrete(dataset)
    # Convert the class column into boolean, 1 for 4(malignant), 0 for 2(benign)
    preprocess.convert_column(dataset, 4, 2)
    random.shuffle(dataset)
    parts = preprocess.create_partitions(TOTAL_PARTITIONS, dataset)
    #for key in parts.keys():
    train, test = utilities.get_train_test_sets(parts, 0)
        # log_reg = twoclass_lr.LG(copy.deepcopy(train), copy.deepcopy(test))
        # log_reg.train_model()
        # log_reg.test_model()
        # print("###")
    # Build the naive bayes likleihood table.
    labels = [i for i in range(2)]
    table = naive_bayes.Model(train, test, labels)
    table.train_model()
    # Test the model.
    table.test_model()
    # # Randomize the dataset
    # random.shuffle(dataset)
    # # Set the default wwights
    # winnow_weights = winnow.set_default_weights(dataset[0][1:])
    # # Split the data set into a train and test set
    # train, test = preprocess.split_list(dataset)
    # # Train the model.
    # winnow_weights = winnow.train(winnow_weights, train)
    # # Test the model.
    # winnow.test(winnow_weights,test)
    # # Build the naive bays likleihood table
    # naive_table = naive_bayes.build_table(train)
    # # Test the model out
    # naive_bayes.test(test, naive_table)

'''
This method executes the glass dataset test.
'''
def do_glass():
    print("The glass dataset will be processed and tested.")
    # Read the data for the specified file
    dataset = preprocess.read_file("./glass.data")
    dataset = utilities.transpose(dataset)
    del dataset[0]
    dataset = utilities.transpose(dataset)

    for entry in range(len(dataset)):
        if utilities.is_float(dataset[entry][0]):
            class_val = dataset[entry][-1]
            dataset[entry] = [float(s.replace(',', '')) for s in dataset[entry][:-1]]
            preprocess.normalize_data(dataset[entry])
            dataset[entry].append(class_val)

    glass_dict = {'1': 0, '2': 1,
                  '3': 2, '5': 3,
                  '6': 4, '7': 5}

    preprocess.build_multiclass(dataset, glass_dict)


    random.shuffle(dataset)
    parts = preprocess.create_partitions(TOTAL_PARTITIONS, dataset)
    for key in parts.keys():
        train, test = utilities.get_train_test_sets(parts, key)
        test = multiclass_lr.LR(train, test, 6)
        test.train_model()
        test.test_model()

'''
This method executes the soybean dataset test.
'''
def do_soy():
    print("The soybean dataset will be processed and tested.")
    # Read the data for the specified file
    dataset = preprocess.read_file("soybean-small.data")
    soy_dict = {'D1': 0, 'D2': 1,
                'D3': 2, 'D4': 3}
    # Convert the columns into their discretized versions.
    preprocess.build_multiclass(dataset, soy_dict)
    # Obtain column representation
    dataset = utilities.transpose(dataset)
    # Iterate through each column and fill in any missing entries
    # and convert the string to integers
    for entry in range(len(dataset)):
        preprocess.fill_column(dataset[entry])
        dataset[entry] = [int(i) for i in dataset[entry]]
        if entry != len(dataset) - 1:
            preprocess.normalize_data(dataset[entry])
    # Obtain row representation
    dataset = utilities.transpose(dataset)
    # Randomize the dataset
    random.shuffle(dataset)
    parts = preprocess.create_partitions(TOTAL_PARTITIONS, dataset)
    for key in parts.keys():
        train, test = utilities.get_train_test_sets(parts, key)
        test = multiclass_lr.LR(train, test, 4)
        test.train_model()
        test.test_model()

#do_soy()
#do_glass()
do_breast_cancer()
#do_house_votes()
#do_iris()
