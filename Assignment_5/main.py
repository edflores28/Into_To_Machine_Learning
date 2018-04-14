import adaline_network
import copy
import preprocess
import utilities
import random


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
    print("Creating a total of", TOTAL_PARTITIONS, "partitions")
    print("Using the logistic regression model")
    # Create the partitions
    parts = preprocess.create_partitions(TOTAL_PARTITIONS, dataset)
    # Obtain the error for logistic regression
    log_error = utilities.two_class_lr_test(parts)
    print("The logistic regression error rate is:", log_error/TOTAL_PARTITIONS, "\n")
    print("Using the Naive Bayes Model")
    # Build the total labels
    labels = [i for i in range(2)]
    # Obtain the error for naive bayes
    naive_error = utilities.naive_test(parts, labels)
    print("The naive bayes error rate is:", naive_error/TOTAL_PARTITIONS, "\n")
    adaline_error = utilities.adaline_test(parts, 2)
    print("The adaline network error rate is:", adaline_error/TOTAL_PARTITIONS, "\n")


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
    # Digitize the data set
    dataset = preprocess.build_discrete(dataset)
    # Shuffle the data set
    random.shuffle(dataset)
    # Copy data for the adaline_network
    adaline_data = copy.deepcopy(dataset)
    print("Creating a total of", TOTAL_PARTITIONS, "partitions")
    print("Using the logistic regression model")
    # Create the partitions
    parts = preprocess.create_partitions(TOTAL_PARTITIONS, dataset)
    # Obtain the error for logistic regression
    log_error = utilities.multi_class_lr_test(parts, 3)
    print("The logistic regression error rate is:", log_error/TOTAL_PARTITIONS, "\n")
    print("Using the Naive Bayes Model")
    # Build the total labels
    labels = [i for i in range(3)]
    # Obtain the error for naive bayes
    naive_error = utilities.naive_test(parts, labels)
    print("The naive bayes error rate is:", naive_error/TOTAL_PARTITIONS, "\n")
    # Obtain column representation
    adaline_data = utilities.transpose(adaline_data)
    # Normalize the data
    for column in adaline_data[:-1]:
        preprocess.normalize_data(column)
    # Obtain the row representation
    adaline_data = utilities.transpose(adaline_data)
    # Create a new partition for adaline
    parts = preprocess.create_partitions(TOTAL_PARTITIONS, adaline_data)
    adaline_error = utilities.adaline_test(parts, 3)
    print("The adaline network error rate is:", adaline_error/TOTAL_PARTITIONS, "\n")


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
    # Obtain row representation
    dataset = utilities.transpose(dataset)
    # Digitize the dataset
    dataset = preprocess.build_discrete(dataset)
    # Convert the class column into boolean, 1 malignant, 0 for benign
    preprocess.convert_column(dataset, '4', '2')
    # Randomly shuffle the data set
    random.shuffle(dataset)
    # Copy data for the adaline_network
    adaline_data = copy.deepcopy(dataset)
    # Create partitions of the data set.
    parts = preprocess.create_partitions(TOTAL_PARTITIONS, dataset)
    print("Creating a total of", TOTAL_PARTITIONS, "partitions")
    print("Using the logistic regression model")
    # Obtain the error for logistic regression
    log_error = utilities.two_class_lr_test(parts)
    print("The logistic regression error rate is:", log_error/TOTAL_PARTITIONS, "\n")
    print("Using the Naive Bayes Model")
    # Build the total labels
    labels = [i for i in range(2)]
    # Obtain the error for naive bayes
    naive_error = utilities.naive_test(parts, labels)
    print("The naive bayes error rate is:", naive_error/TOTAL_PARTITIONS, "\n")
    # Obtain column representation
    adaline_data = utilities.transpose(adaline_data)
    # Normalize the data
    for column in adaline_data[:-1]:
        preprocess.normalize_data(column)
    # Obtain the row representation
    adaline_data = utilities.transpose(adaline_data)
    # Create a new partition for adaline
    parts = preprocess.create_partitions(TOTAL_PARTITIONS, adaline_data)
    adaline_error = utilities.adaline_test(parts, 2)
    print("The adaline network error rate is:", adaline_error/TOTAL_PARTITIONS, "\n")


'''
This method executes the glass dataset test.
'''
def do_glass():
    print("The glass dataset will be processed and tested.\n")
    # Read the data for the specified file
    dataset = preprocess.read_file("./glass.data")
    # Obtain column representation
    dataset = utilities.transpose(dataset)
    # Delete index 0 since this is the ID field and is not needed
    del dataset[0]
    # Obtain row representation
    dataset = utilities.transpose(dataset)
    # Create a dictionary for the class labels
    glass_dict = {'1': 0, '2': 1,
                  '3': 2, '5': 3,
                  '6': 4, '7': 5}
    # Update the class labels according to the dictionary
    preprocess.build_multiclass(dataset, glass_dict)
    # Digitize the data set
    dataset = preprocess.build_discrete(dataset)
    # Shuffle the data set
    random.shuffle(dataset)
    # Copy data for the adaline_network
    adaline_data = copy.deepcopy(dataset)
    print("Creating a total of", TOTAL_PARTITIONS, "partitions")
    print("Using the logistic regression model")
    # Create the partitions
    parts = preprocess.create_partitions(TOTAL_PARTITIONS, dataset)
    # Obtain the error for logistic regression
    log_error = utilities.multi_class_lr_test(parts, 6)
    print("The logistic regression error rate is:", log_error/TOTAL_PARTITIONS, "\n")
    print("Using the Naive Bayes Model")
    # Build the total labels
    labels = [i for i in range(6)]
    # Obtain the error for naive bayes
    naive_error = utilities.naive_test(parts, labels)
    print("The naive bayes error rate is:", naive_error/TOTAL_PARTITIONS, "\n")
    # Obtain column representation
    adaline_data = utilities.transpose(adaline_data)
    # Normalize the data
    for column in adaline_data[:-1]:
        preprocess.normalize_data(column)
    # Obtain the row representation
    adaline_data = utilities.transpose(adaline_data)
    # Create a new partition for adaline
    parts = preprocess.create_partitions(TOTAL_PARTITIONS, adaline_data)
    adaline_error = utilities.adaline_test(parts, 6)
    print("The adaline network error rate is:", adaline_error/TOTAL_PARTITIONS, "\n")


'''
This method executes the soybean dataset test.
'''
def do_soy():
    print("The soybean dataset will be processed and tested.\n")
    # Read the data for the specified file
    dataset = preprocess.read_file("./soybean-small.data")
    # Create a dictionary for the class labels
    soy_dict = {'D1': 0, 'D2': 1,
                'D3': 2, 'D4': 3}
    # Convert the columns into their discretized versions.
    preprocess.build_multiclass(dataset, soy_dict)
    # Digitize the data set
    dataset = preprocess.build_discrete(dataset)
    # Randomize the dataset
    random.shuffle(dataset)
    # Copy data for the adaline_network
    adaline_data = copy.deepcopy(dataset)
    print("Creating a total of", TOTAL_PARTITIONS, "partitions")
    print("Using the logistic regression model")
    # Create the partitions
    parts = preprocess.create_partitions(TOTAL_PARTITIONS, dataset)
    # Obtain the error for logistic regression
    log_error = utilities.multi_class_lr_test(parts, 4)
    print("The logistic regression error rate is:", log_error/TOTAL_PARTITIONS, "\n")
    print("Using the Naive Bayes Model")
    # Build the total labels
    labels = [i for i in range(4)]
    # Obtain the error for naive bayes
    naive_error = utilities.naive_test(parts, labels)
    print("The naive bayes error rate is:", naive_error/TOTAL_PARTITIONS, "\n")
    # Obtain column representation
    adaline_data = utilities.transpose(adaline_data)
    # Normalize the data
    for column in adaline_data[:-1]:
        preprocess.normalize_data(column)
    # Obtain the row representation
    adaline_data = utilities.transpose(adaline_data)
    # Create a new partition for adaline
    parts = preprocess.create_partitions(TOTAL_PARTITIONS, adaline_data)
    adaline_error = utilities.adaline_test(parts, 4)
    print("The adaline network error rate is:", adaline_error/TOTAL_PARTITIONS, "\n")

#do_soy()
#do_glass()
#do_breast_cancer()
do_house_votes()
#do_iris()
