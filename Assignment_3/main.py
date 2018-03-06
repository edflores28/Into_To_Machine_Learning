import preprocess
import nearest_neighbor

def do_ecoli():
    remove_class = ["omL","imL","imS"]
    dataset = preprocess.read_file(filename="ecoli.data",rem_list=remove_class,class_idx=8)
    zero_col_dict = {}
    counter = 0
    for entry in range(len(dataset)):
        if dataset[entry][0] not in zero_col_dict:
            counter += 1
            zero_col_dict[dataset[entry][0]] = counter
        dataset[entry][0] = str(zero_col_dict[dataset[entry][0]])
        last = dataset[entry][-1]
        dataset[entry] = [float(s.replace(',','')) for s in dataset[entry][:-1]]
        dataset[entry].append(last)

    partition_dict = preprocess.create_partitions(5,8,dataset)
    train = []
    test = []
    for key in partition_dict:
        if key == 1:
            test += partition_dict[key]
        else:
            train += partition_dict[key]

    #print(nearest_neighbor.kNN(dataset,train,test[0],8,3))
    nearest_neighbor.cNN(dataset,train,8)
do_ecoli()
