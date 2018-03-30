import preprocess
import utilities
import id3
import prune


# Obtain the dataset
dataset = preprocess.read_file(filename="./abalone.data", split=',')
# Obtain a column representation
dataset = utilities.transpose(dataset)
# For continous data convert them from strings to float
for entry in range(len(dataset)):
    if utilities.is_float(dataset[entry][0]):
        dataset[entry] = [float(s.replace(',', '')) for s in dataset[entry]]
# Obtain the row representation
dataset = utilities.transpose(dataset)
train_len = int(0.9*len(dataset))
test_len = len(dataset)-train_len
train = dataset[:train_len]
test = dataset[train_len:]
feature_indices = [i for i in range(len(dataset[0]))]
x = id3.ID3(train, feature_indices)
y = x.build_tree()

test = prune.Prune(y, test)
test.reduced_error_prune(y)
