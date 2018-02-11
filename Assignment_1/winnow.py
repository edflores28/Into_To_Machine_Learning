import preprocess

# Constant variables
MAX_ROWS_PRINT = 10
MULTI_ROWS_PRINT = 5
PRINT_COUNTER = 0;
THRESHOLD = 0.75
ALPHA = 2.0

'''
This method sets the default weight for the
data set inputs
'''
def set_default_weights(data):
    temp = []
    for i in range(len(data)):
        temp.append(1.0)
    return temp

'''
This method calculates a new weight for the
inputs when they need to be promoted
'''
def promote(weights,row):
    global PRINT_COUNTER
    for index in range(1, len(row)):
        if row[index] == 1:
            weights[index-1] = weights[index-1]*ALPHA
    if PRINT_COUNTER < MAX_ROWS_PRINT:
        print ("Weights were promoted, new weights are: ")
        print (weights,"\n")
    return weights

'''
This method calculates a new weight for the
inputs when they need to be demoted
'''
def demote(weights,row):
    global PRINT_COUNTER
    for index in range(1, len(row)):
        if row[index] == 1:
            weights[index-1] = weights[index-1]/ALPHA
    if PRINT_COUNTER < MAX_ROWS_PRINT:
        print ("Weights were demoted, new weights are: ")
        print (weights,"\n")
    return weights

'''
This method predicts the result based on the weights
'''
def prediction(weights,row):
    result = 0.0
    for index in range(1, len(row)):
        result += row[index]*weights[index-1]
    return 1 if result >= THRESHOLD else 0

'''
This method iterates through the training list
and predicts the result. Based on the results
the weights are recalculated based on promotion
and demotion.
'''
def train(weights, train_list):
    global PRINT_COUNTER
    print ("Staring to train the winnow model. All weights are 1.0")
    for index in range(len(train_list)):
        predict = prediction(weights,train_list[index])
        if PRINT_COUNTER < MAX_ROWS_PRINT:
            print ("Predicted: ", predict, " for the following inputs: ")
            print (train_list[index], "\n")
        if predict == 1 and train_list[index][0] == 0:
            weights = demote(weights, train_list[index])
        if predict == 0 and train_list[index][0] == 1:
            weights = promote(weights, train_list[index])
        if PRINT_COUNTER < MAX_ROWS_PRINT:
            PRINT_COUNTER += 1
    return weights

'''
This method iterates through the test list
and predicts the result. Counters are increment
to keep track of right and wrong answers.
'''
def test(weights,test_list):
    print("Starting winnow test")
    results = [0,0]
    print("Weights to be used to test: ")
    print (weights,"\n")
    for index in range(len(test_list)):
        result = prediction(weights,test_list[index])
        if result == test_list[index][0]:
            results[1] += 1
        else:
            results[0] += 1
    print("\nWinnow Statistics:")
    print("Total matched:\t\t", results[1])
    print("Total matched:\t\t", results[0])
    print("Percentage matched:\t", (results[1]*100)/(results[1]+results[0]),"\n")

'''
This method iterates through the test list
and predicts the result. Counters are increment
to keep track of right and wrong answers. Since
this method is for multi classifications the
weights used are based on the class.
'''
def multi_test(weights,test_list, data_dict):
    print("Starting winnow multiclass test")
    res = [[0,0] for i in range(len(weights))]
    key_list = list(data_dict.values())
    for key in data_dict:
        print("Weight to use for class", key, "weights")
        print(weights[key_list.index(data_dict[key])],"\n")
    for index in range(len(test_list)):
        index = key_list.index(test_list[index][:len(weights)])
        result = prediction(weights[index],test_list[index])
        if result == test_list[index][0]:
            res[index][1] += 1
        else:
            res[index][0] += 1
    match = 0
    no_match = 0
    # Add all the matches and no matches
    for index in range(len(res)):
        match += res[index][1]
        no_match += res[index][0]
    print("MultiClass Winnow Statistics:")
    print("Total matched:\t\t", match)
    print("Total not matched:\t", no_match)
    print("Percentage matched:\t", (match*100)/(match+no_match),"\n")

def reset_counter():
    global PRINT_COUNTER
    PRINT_COUNTER = 0
