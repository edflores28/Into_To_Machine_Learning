import preprocess
import random
import backpropagation
import utilities

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
    prop = backpropagation.OneLayer(train, test, 2, 1)
    prop.train_model()


do_house_votes()
