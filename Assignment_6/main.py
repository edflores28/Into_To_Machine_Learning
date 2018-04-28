import preprocess
import random
import utilities
import argparse

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
    # Create partitions
    parts = preprocess.create_partitions(TOTAL_PARTITIONS, dataset)
    print("Partitions created!")
    # Execute the test_list
    utilities.main_test(parts, 16, [19, 7], 1)


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
    # Create partitions
    parts = preprocess.create_partitions(TOTAL_PARTITIONS, dataset)
    print("Partitions created!")
    # Execute the test_list
    utilities.main_test(parts, 16, [19, 7], 1)


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
    # Create partitions
    parts = preprocess.create_partitions(TOTAL_PARTITIONS, dataset)
    print("Partitions created!")
    # Execute the test_list
    utilities.main_test(parts, 16, [19, 7], 3)


def do_glass():
    '''
    This method executes the glass dataset test.
    '''
    print("The glass dataset will be processed and tested.\n")
    # Read the data for the specified file
    dataset = preprocess.read_file("./glass.data")
    # Obtain column representation
    dataset = utilities.transpose(dataset)
    # Delete index 0 since this is the ID field and is not needed
    del dataset[0]
    # Normalize the data
    for column in dataset[:-1]:
        preprocess.normalize_data(column)
    # Obtain row representation
    dataset = utilities.transpose(dataset)
    # Create a dictionary for the class labels
    glass_dict = {'1': 0, '2': 1,
                  '3': 2, '5': 3,
                  '6': 4, '7': 5}
    # Update the class labels according to the dictionary
    preprocess.build_multiclass(dataset, glass_dict)
    # Shuffle the data set
    random.shuffle(dataset)
    # Create partitions
    parts = preprocess.create_partitions(TOTAL_PARTITIONS, dataset)
    print("Partitions created!")
    # Execute the test_list
    utilities.main_test(parts, 30, [23, 11], 6)


def do_soy():
    '''
    This method executes the soybean dataset test.
    '''
    print("The soybean dataset will be processed and tested.\n")
    # Read the data for the specified file
    dataset = preprocess.read_file("./soybean-small.data")
    # Create a dictionary for the class labels
    soy_dict = {'D1': 0, 'D2': 1,
                'D3': 2, 'D4': 3}
    # Convert the columns into their discretized versions.
    preprocess.build_multiclass(dataset, soy_dict)
    # Obtain column representation
    dataset = utilities.transpose(dataset)
    # Normalize the data
    for column in dataset[:-1]:
        preprocess.normalize_data(column)
    # Obtain the row representation
    dataset = utilities.transpose(dataset)
    # Randomize the dataset
    random.shuffle(dataset)
    # Create partitions
    parts = preprocess.create_partitions(TOTAL_PARTITIONS, dataset)
    print("Partitions created!")
    # Execute the test_list
    utilities.main_test(parts, 11, [22, 11], 4)

'''
MAIN APPLICATION
'''
# Create a parser for the command line arguments
parser = argparse.ArgumentParser(description="Intro to ML Project 6")
parser.add_argument('-v', action="store_true", default=False, help='Execute house votes test')
parser.add_argument('-s', action="store_true", default=False, help='Execute soybean test')
parser.add_argument('-b', action="store_true", default=False, help='Execute breast cancer test')
parser.add_argument('-i', action="store_true", default=False, help='Execute iris test')
parser.add_argument('-g', action="store_true", default=False, help='Execute glass test')

results = parser.parse_args()

# Perform the tests based on the input
if results.v:
    do_house_votes()
if results.s:
    do_soy()
if results.b:
    do_breast_cancer()
if results.i:
    do_iris()
if results.g:
    do_glass()
