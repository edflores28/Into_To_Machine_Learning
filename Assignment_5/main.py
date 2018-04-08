import copy
import preprocess
import logistic_regression
import utilities

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
    parts = preprocess.create_partitions(total_parts, dataset)
    for key in parts.keys():
        train, test = utilities.get_train_test_sets(parts, key)
        log_reg = logistic_regression.LG(copy.deepcopy(train), copy.deepcopy(test))
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
    dataset = utilities.transpose(dataset)

    iris_dict = {'Iris-virginica' : 0,
                 'Iris-versicolor': 1,
                 'Iris-setosa'    : 2}
    preprocess.build_multiclass(dataset,iris_dict)

    # Convert the real data present in the dataset and diiscretize them.
    # dataset = preprocess.build_discrete(dataset)
    #
    #
    # # Build the naive bayes likleihood table.
    # naive_table = naive_bayes.build_table(train)
    # # Test the model.
    # naive_bayes.test(test, naive_table)
#do_house_votes()
do_iris()
