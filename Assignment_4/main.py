import preprocess
import utilities
import id3


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
dataset = dataset[:20]
feature_indices = [i for i in range(len(dataset[0]))]
x = id3.ID3(dataset, feature_indices, 0)
x.build_tree()
