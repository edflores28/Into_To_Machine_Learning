import preprocess

THRESHOLD = 0.75
ALPHA = 2.0

def set_default_weights(data):
    temp = []
    for i in range(len(data)):
        temp.append(1.0)
    return temp

def promote(weights,row):
    for i in range(1, len(row)):
        if row[i] == 1:
            weights[i-1] = weights[i-1]*ALPHA
    return weights

def demote(weights,row):
    for i in range(1, len(row)):
        if row[i] == 1:
            weights[i-1] = weights[i-1]/ALPHA
    return weights

def prediction(weights,row):
    result = 0.0
    for i in range(1, len(row)):
        result += row[i]*weights[i-1]
    return 1 if result >= THRESHOLD else 0

def train(weights, train_list):
    for i in range(len(train_list)):
        predict = prediction(weights,train_list[i])
        if predict == 1 and train_list[i][0] == 0:
            weights = demote(weights, train_list[i])
        if predict == 0 and train_list[i][0] == 1:
            weights = promote(weights, train_list[i])
    return weights

def test(weights,train_list):
    results = [0,0]
    for i in range(len(train_list)):
        result = prediction(weights,train_list[i])
        if result == train_list[i][0]:
            results[1] += 1
        else:
            results[0] += 1
    print("Winnow Statistics:")
    print("Total successfully matched:\t", results[1])
    print("Total successfully not matched:\t", results[0])
    print("Percentage matched:\t", (results[1]*100)/(results[1]+results[0]),"\n")

def multi_test(weights,train_list, data_dict):
    res = [[0,0] for i in range(len(weights))]

    key_list = list(data_dict.values())

    for i in range(len(train_list)):
        index = key_list.index(train_list[i][:len(weights)])
        result = prediction(weights[index],train_list[i])
        if result == train_list[i][0]:
            res[index][1] += 1
        else:
            res[index][0] += 1

    match = 0
    no_match = 0
    for i in range(len(res)):
        match += res[i][1]
        no_match += res[i][0]

    print("MultiClass Winnow Statistics:")
    print("Total successfully matched:\t", match)
    print("Total successfully not matched:\t", no_match)
    print("Percentage matched:\t", (match*100)/(match+no_match),"\n")
