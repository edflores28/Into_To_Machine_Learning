import winnow
import naive_bayes
import preprocess
import random
import argparse

# Constant values
iris_dict = {'Iris-virginica' : [0,0,1],
             'Iris-versicolor': [0,1,0],
             'Iris-setosa'    : [1,0,0]}

soy_dict = {'D1': [0,0,0,1], 'D2': [0,0,1,0],
            'D3': [0,1,0,0], 'D4': [1,0,0,0]}

glass_dict = {'1': [0,0,0,0,0,1], '2': [0,0,0,0,1,0],
              '3': [0,0,0,1,0,0], '5': [0,0,1,0,0,0],
              '6': [0,1,0,0,0,0], '7': [1,0,0,0,0,0]}

'''
This method executes the house votes dataset test.
'''
def do_house_votes():
    print("The house votes dataset will be processed and tested.")
    # Read the data for the specified file
    dataset = preprocess.read_file("house-votes-84.data")
    # Convert the house data
    preprocess.convert_house_data(dataset)
    # Randomize the dataset
    random.shuffle(dataset)
    # Set the default house_weights
    winnow_weights = winnow.set_default_weights(dataset[0][1:]);
    # Split the list into a training and test list
    train, test = preprocess.split_list(dataset)
    # Train for winnow
    winnow_weights = winnow.train(winnow_weights,train)
    # Test the model out.
    winnow.test(winnow_weights,test)
    # Train for naive bayes
    naive_table = naive_bayes.build_table(train)
    # Test the model out
    naive_bayes.test(test, naive_table)

'''
This method executes the breast cancer dataset test.
'''
def do_breast_cancer():
    print("The breast dataset will be processed and tested.")
    # Read the data for the specified file
    dataset = preprocess.read_file("breast-cancer-wisconsin.data")
    # Swap the specified columns in order to have the classifications
    # in the first column. This will also delete the Id column.
    dataset = preprocess.swap_columns(0,10,dataset)
    # Convert the real data present in the dataset and diiscretize them.
    dataset = preprocess.build_discrete(dataset)
    # Convert the class column into boolean, 1 for 4(malignant), 0 for 2(benign)
    preprocess.convert_column(dataset,4,2)
    # Randomize the dataset
    random.shuffle(dataset)
    # Set the default wwights
    winnow_weights = winnow.set_default_weights(dataset[0][1:])
    # Split the data set into a train and test set
    train, test = preprocess.split_list(dataset)
    # Train the model.
    winnow_weights = winnow.train(winnow_weights, train)
    # Test the model.
    winnow.test(winnow_weights,test)
    # Build the naive bays likleihood table
    naive_table = naive_bayes.build_table(train)
    # Test the model out
    naive_bayes.test(test, naive_table)

'''
This method executes the iris dataset test.
'''
def do_iris():
    print("The iris dataset will be processed and tested.")
    # Read the data for the specified file
    dataset = preprocess.read_file("iris.data")
    # Swap the specified columns in order to have the classifications
    # in the first column.
    dataset = preprocess.swap_columns(0,4,dataset,'False')
    # Convert the columns into their discretized versions.
    preprocess.build_multiclass(dataset,iris_dict)
    # Convert the real data present in the dataset and diiscretize them.
    dataset = preprocess.build_discrete(dataset)
    # Randomize the dataset
    random.shuffle(dataset)
    # Set the default weights for the classifications
    setosa_weights = winnow.set_default_weights(dataset[0][1:]);
    versi_weights = winnow.set_default_weights(dataset[0][1:]);
    virg_weights = winnow.set_default_weights(dataset[0][1:]);
    # Split the data set into a train and test set
    train, test = preprocess.split_list(dataset)
    # The following processing will train all the weights sets
    # for each classification. The weights will be generated, then
    # the column will be swapped with the column next to it.
    # Once the last classification is complete the columns will
    # be swapped once more to get the initial state of the dataset.
    setosa_weights = winnow.train(setosa_weights, train)
    dataset = preprocess.swap_columns(0,1,dataset,'False')
    versi_weights = winnow.train(versi_weights, train)
    dataset = preprocess.swap_columns(1,2,dataset,'False')
    virg_weights = winnow.train(virg_weights, train)
    dataset = preprocess.swap_columns(2,1,dataset,'False')
    # Pack all the weights into a list and test the model.
    multi_weights = [virg_weights,versi_weights,setosa_weights]
    winnow.multi_test(multi_weights,test, iris_dict)
    # Build the naive bayes likleihood table.
    naive_table = naive_bayes.build_table(train)
    # Test the model.
    naive_bayes.test(test, naive_table)

'''
This method executes the soybean dataset test.
'''
def do_soy():
    print("The soybean dataset will be processed and tested.")
    # Read the data for the specified file
    dataset = preprocess.read_file("soybean-small.data")
    # Swap the specified columns in order to have the classifications
    # in the first column.
    dataset = preprocess.swap_columns(0,35,dataset,'False')
    # Convert the columns into their discretized versions.
    preprocess.build_multiclass(dataset,soy_dict)
    # Convert the real data present in the dataset and diiscretize them.
    dataset = preprocess.build_discrete(dataset)
    # Randomize the dataset
    random.shuffle(dataset)
    # Set the default weights for all the classifications
    d1_weights = winnow.set_default_weights(dataset[0][1:]);
    d2_weights = winnow.set_default_weights(dataset[0][1:]);
    d3_weights = winnow.set_default_weights(dataset[0][1:]);
    d4_weights = winnow.set_default_weights(dataset[0][1:]);
    # Split the data set into a train and test set
    train, test = preprocess.split_list(dataset)
    # The following processing will train all the weights sets
    # for each classification. The weights will be generated, then
    # the column will be swapped with the column next to it.
    # Once the last classification is complete the columns will
    # be swapped once more to get the initial state of the dataset.
    d1_weights = winnow.train(d1_weights, train)
    dataset = preprocess.swap_columns(0,1,dataset,'False')
    d2_weights = winnow.train(d2_weights, train)
    dataset = preprocess.swap_columns(1,2,dataset,'False')
    d3_weights = winnow.train(d3_weights, train)
    dataset = preprocess.swap_columns(2,3,dataset,'False')
    d4_weights = winnow.train(d4_weights, train)
    dataset = preprocess.swap_columns(3,1,dataset,'False')
    # Pack all the weights into a list and test the model.
    multi_weights = [d1_weights,d2_weights,d3_weights,d4_weights]
    winnow.multi_test(multi_weights,test, soy_dict)
    # Build the naive bayes likleihood table.
    naive_table = naive_bayes.build_table(train)
    # Test the model.
    naive_bayes.test(test, naive_table)

'''
This method executes the glass dataset test.
'''
def do_glass():
    print("The glass dataset will be processed and tested.")
    # Read the data for the specified file
    dataset = preprocess.read_file("glass.data")
    dataset = preprocess.swap_columns(0,10,dataset)

    preprocess.build_multiclass(dataset,glass_dict)
    # Convert the real data present in the dataset and diiscretize them.
    dataset = preprocess.build_discrete(dataset)
    # Randomize the dataset
    random.shuffle(dataset)
    # Set the default weights for all the classifications
    x1_weights = winnow.set_default_weights(dataset[0][1:]);
    x2_weights = winnow.set_default_weights(dataset[0][1:]);
    x3_weights = winnow.set_default_weights(dataset[0][1:]);
    x4_weights = winnow.set_default_weights(dataset[0][1:]);
    x5_weights = winnow.set_default_weights(dataset[0][1:]);
    x6_weights = winnow.set_default_weights(dataset[0][1:]);
    # Split the data set into a train and test set
    train, test = preprocess.split_list(dataset)
    # The following processing will train all the weights sets
    # for each classification. The weights will be generated, then
    # the column will be swapped with the column next to it.
    # Once the last classification is complete the columns will
    # be swapped once more to get the initial state of the dataset.
    x1_weights = winnow.train(x1_weights, train)
    dataset = preprocess.swap_columns(0,1,dataset,'False')
    x2_weights = winnow.train(x2_weights, train)
    dataset = preprocess.swap_columns(1,2,dataset,'False')
    x3_weights = winnow.train(x3_weights, train)
    dataset = preprocess.swap_columns(2,3,dataset,'False')
    x4_weights = winnow.train(x4_weights, train)
    dataset = preprocess.swap_columns(3,4,dataset,'False')
    x5_weights = winnow.train(x5_weights, train)
    dataset = preprocess.swap_columns(4,5,dataset,'False')
    x6_weights = winnow.train(x6_weights, train)
    dataset = preprocess.swap_columns(5,0,dataset,'False')
    # Pack all the weights into a list and test the model.
    multi_weights = [x1_weights,x2_weights,x3_weights,x4_weights,x5_weights,x6_weights]
    winnow.multi_test(multi_weights,test, glass_dict)
    # Build the naive bayes likleihood table.
    naive_table = naive_bayes.build_table(train)
    # Test the model.
    naive_bayes.test(test, naive_table)

# Create a parser for the command line arguments
parser = argparse.ArgumentParser(description="Intro to ML Project 1")
parser.add_argument('-v',action="store_true", default=False, help='Execute house votes test')
parser.add_argument('-s',action="store_true", default=False, help='Execute soybean test')
parser.add_argument('-b',action="store_true", default=False, help='Execute breast cancer test')
parser.add_argument('-i',action="store_true", default=False, help='Execute iris test')
parser.add_argument('-g',action="store_true", default=False, help='Execute glass test')

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

print("Tests executed.")
