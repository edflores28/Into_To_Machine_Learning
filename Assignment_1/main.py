import winnow
import naive_bayes
import preprocess
import random

iris_dict = {'Iris-virginica' : [0,0,1],
             'Iris-versicolor': [0,1,0],
             'Iris-setosa'    : [1,0,0]}

soy_dict = {'D1': [0,0,0,1], 'D2': [0,0,1,0],
            'D3': [0,1,0,0], 'D4': [1,0,0,0]}

glass_dict = {'1': [0,0,0,0,0,1], '2': [0,0,0,0,1,0],
              '3': [0,0,0,1,0,0], '5': [0,0,1,0,0,0],
              '6': [0,1,0,0,0,0], '7': [1,0,0,0,0,0]}

# Read the data for the specified file
dataset = preprocess.read_file("house-votes-84.data")
# Convert the house data
preprocess.convert_house_data(dataset)
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

dataset = preprocess.read_file("breast-cancer-wisconsin.data")
dataset = preprocess.swap_columns(0,10,dataset)
dataset = preprocess.build_discrete(dataset)
preprocess.convert_column(dataset,4,2)
winnow_weights = winnow.set_default_weights(dataset[0][1:])
train, test = preprocess.split_list(dataset)
winnow_weights = winnow.train(winnow_weights, train)
winnow.test(winnow_weights,test)

naive_table = naive_bayes.build_table(train)
# Test the model out
naive_bayes.test(test, naive_table)

dataset = preprocess.read_file("iris.data")
dataset = preprocess.swap_columns(0,4,dataset,'False')
preprocess.build_multiclass(dataset,iris_dict)
dataset = preprocess.build_discrete(dataset)
random.shuffle(dataset)

setosa_weights = winnow.set_default_weights(dataset[0][1:]);
versi_weights = winnow.set_default_weights(dataset[0][1:]);
virg_weights = winnow.set_default_weights(dataset[0][1:]);

train, test = preprocess.split_list(dataset)

setosa_weights = winnow.train(setosa_weights, train)
dataset = preprocess.swap_columns(0,1,dataset,'False')
versi_weights = winnow.train(versi_weights, train)
dataset = preprocess.swap_columns(1,2,dataset,'False')
virg_weights = winnow.train(virg_weights, train)
dataset = preprocess.swap_columns(2,1,dataset,'False')

multi_weights = [virg_weights,versi_weights,setosa_weights]
winnow.multi_test(multi_weights,test, iris_dict)

naive_table = naive_bayes.build_table(train)
naive_bayes.test(test, naive_table)
##########################################################################
dataset = preprocess.read_file("soybean-small.data")
dataset = preprocess.swap_columns(0,35,dataset,'False')

preprocess.build_multiclass(dataset,soy_dict)
dataset = preprocess.build_discrete(dataset)
#random.shuffle(dataset)

d1_weights = winnow.set_default_weights(dataset[0][1:]);
d2_weights = winnow.set_default_weights(dataset[0][1:]);
d3_weights = winnow.set_default_weights(dataset[0][1:]);
d4_weights = winnow.set_default_weights(dataset[0][1:]);

train, test = preprocess.split_list(dataset)

d1_weights = winnow.train(d1_weights, train)
dataset = preprocess.swap_columns(0,1,dataset,'False')
d2_weights = winnow.train(d2_weights, train)
dataset = preprocess.swap_columns(1,2,dataset,'False')
d3_weights = winnow.train(d3_weights, train)
dataset = preprocess.swap_columns(2,3,dataset,'False')
d4_weights = winnow.train(d4_weights, train)
dataset = preprocess.swap_columns(3,1,dataset,'False')

multi_weights = [d1_weights,d2_weights,d3_weights,d4_weights]
winnow.multi_test(multi_weights,test, soy_dict)

naive_table = naive_bayes.build_table(train)
naive_bayes.test(test, naive_table)
##############################################################################
dataset = preprocess.read_file("glass.data")
dataset = preprocess.swap_columns(0,10,dataset)

preprocess.build_multiclass(dataset,glass_dict)
dataset = preprocess.build_discrete(dataset)
random.shuffle(dataset)

x1_weights = winnow.set_default_weights(dataset[0][1:]);
x2_weights = winnow.set_default_weights(dataset[0][1:]);
x3_weights = winnow.set_default_weights(dataset[0][1:]);
x4_weights = winnow.set_default_weights(dataset[0][1:]);
x5_weights = winnow.set_default_weights(dataset[0][1:]);
x6_weights = winnow.set_default_weights(dataset[0][1:]);

train, test = preprocess.split_list(dataset)

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

multi_weights = [x1_weights,x2_weights,x3_weights,x4_weights,x5_weights,x6_weights]
winnow.multi_test(multi_weights,test, glass_dict)

naive_table = naive_bayes.build_table(train)
naive_bayes.test(test, naive_table)
