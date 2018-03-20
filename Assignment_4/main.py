import preprocess
import utilities
import id3

# Label which features are categorical or not
is_continuous = [False, True, True, True, True, True, True, True, False]
# Obtain the dataset
dataset = preprocess.read_file(filename="./abalone.data", split=',')
# Obtain a column representation
dataset = utilities.transpose(dataset)
# For continous data convert them from strings to float
for entry in range(len(dataset)):
    if is_continuous[entry]:
        dataset[entry] = [float(s.replace(',', '')) for s in dataset[entry]]
# Obtain the row representation
dataset = utilities.transpose(dataset)
dataset = dataset[:20]
x = id3.ID3(dataset, is_continuous)
x.build_tree()
