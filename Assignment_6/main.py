import preprocess
import random
import backpropagation_zero
import backpropagation_one
import backpropagation_two
import utilities
import copy

TOTAL_PARTITIONS = 5


def do_house_votes():
    '''
    This method executes the house votes dataset test.
    '''
    print("The house votes dataset will be processed and tested.\n")
    # Read the data for the specified file
    dataset = preprocess.read_file("./house-votes-84.data")
    # Convert the house data
    preprocess.convert_house_data(dataset)
    # Swap the class column with the last column
    preprocess.swap_columns(0, len(dataset[0])-1, dataset, False)
    # Randomize the dataset
    random.shuffle(dataset)
    # print("Creating a total of", TOTAL_PARTITIONS, "partitions")
    # print("Using the logistic regression model")
    # input("Press Enter to continue...")
    # Create the partitions
    parts = preprocess.create_partitions(TOTAL_PARTITIONS, dataset)
    train, test = utilities.get_train_test_sets(parts, 0)
    prop = backpropagation_one.Model(train, test, 15, 1)
    prop.train_model()
    prop.test_model()


def do_breast_cancer():
    '''
    This method executes the breast cancer dataset test.
    '''
    print("The breast dataset will be processed and tested.\n")
    # Read the data for the specified file
    dataset = preprocess.read_file("./breast-cancer-wisconsin.data")
    # Obtain column representation
    dataset = utilities.transpose(dataset)
    # Delete index 0 since this is the ID field and is not needed
    del dataset[0]
    # Normalize the data minus the classification columns
    for column in range(len(dataset[:-1])):
        preprocess.normalize_data(dataset[column])
    # Obtain row representation
    dataset = utilities.transpose(dataset)
    # Convert the class column into boolean, 1 malignant, 0 for benign
    preprocess.convert_column(dataset, '4', '2')
    # Randomly shuffle the data set
    random.shuffle(dataset)

    parts = preprocess.create_partitions(TOTAL_PARTITIONS, dataset)
    train, test = utilities.get_train_test_sets(parts, 0)
    # prop = backpropagation_one.Model(train, test, 10, 1)
    # prop.train_model()
    # prop.test_model()
    prop = backpropagation_zero.Model(train, test, 1)
    prop.train_model()
    prop.test_model()


def do_iris():
    print("The iris dataset will be processed and tested.\n")
    # Read the data for the specified file
    dataset = preprocess.read_file("./iris.data")
    # Create a dictionary for the class labels
    iris_dict = {'Iris-virginica': 0,
                 'Iris-versicolor': 1,
                 'Iris-setosa': 2}
    # Update the class labels according to the dictionary
    preprocess.build_multiclass(dataset, iris_dict)
    # Obtain column representation
    dataset = utilities.transpose(dataset)
    # Normalize the data
    for column in dataset[:-1]:
        preprocess.normalize_data(column)
    # Obtain the row representation
    dataset = utilities.transpose(dataset)
    # Shuffle the data set
    random.shuffle(dataset)
    parts = preprocess.create_partitions(TOTAL_PARTITIONS, dataset)
    zero_error = 0.0
    one_error = 0.0
    two_error = 0.0
    for key in parts.keys():
        train, test = utilities.get_train_test_sets(parts, key)
        model = backpropagation_zero.Model(copy.deepcopy(train), copy.deepcopy(test), 3)
        model.train_model()
        zero_error += model.test_model()
        model = backpropagation_one.Model(copy.deepcopy(train), copy.deepcopy(test), 19, 3)
        model.train_model()
        one_error += model.test_model()
        model = backpropagation_two.Model(copy.deepcopy(train), copy.deepcopy(test), [6, 10], 3)
        model.train_model()
        two_error += model.test_model()

    print(zero_error/5, one_error/5, two_error/5)
#do_breast_cancer()
#do_house_votes()
do_iris()
