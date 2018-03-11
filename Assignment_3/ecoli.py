import preprocess
import nearest_neighbor
import neural_network
import random
import utilities
import cluster
import time

'''
Perform the tests on the ecoli data set
'''
def do_ecoli():
    remove_class = ["omL","imL","imS"]
    dataset = preprocess.read_file(filename="ecoli.data",rem_list=remove_class,class_idx=8)
    zero_col_dict = {}
    counter = 0
    # Process the data set
    for entry in range(len(dataset)):
        if dataset[entry][0] not in zero_col_dict:
            counter += 0.1
            zero_col_dict[dataset[entry][0]] = counter
        dataset[entry][0] = str(zero_col_dict[dataset[entry][0]])
        last = dataset[entry][-1]
        dataset[entry] = [float(s.replace(',','')) for s in dataset[entry][:-1]]
        dataset[entry].append(last)
    # Obtain partitions of the dataset
    partition_dict = preprocess.create_partitions(5,8,dataset)
    accuracy = 0.0
    k1=4
    k2=4
    k3=4
    clusters = 10
    print("Performing k-nearest neighbor with k =", k1)
    time.sleep(1)
    for key in partition_dict:
        train, test = utilities.get_train_test_sets(partition_dict,key)
        accuracy += nearest_neighbor.kNN_Test(dataset, train, test,len(dataset[0])-1,k1)
    print("Percentage of total correct:",accuracy/5.0,"\n")
    print("Test complete","\n")
    accuracy = 0.0
    input("Press Enter to continue...")
    print("Performing condensed nearest neighbor & k-nearest neighbor with k =", k2)
    time.sleep(1)
    for key in partition_dict:
        train, test = utilities.get_train_test_sets(partition_dict,key)
        train = nearest_neighbor.cNN(dataset, train, len(dataset[0])-1)
        accuracy += nearest_neighbor.kNN_Test(dataset, train, test,len(dataset[0])-1,k2)
    print("Percentage of total correct:",accuracy/5.0,"\n")
    print("Test complete","\n")
    accuracy = 0.0
    print("Performing k means clustering with",clusters,"clusters")
    print("with k-nearest neighbors with k =", k3)
    input("Press Enter to continue...")
    time.sleep(1)
    for key in partition_dict:
        cluster_train = []
        train, test = utilities.get_train_test_sets(partition_dict,key)
        for entry in train:
            cluster_train.append(dataset[entry])
        #Convert the train test into cluster representation
        # Transpose the dataset and remove the class columns
        temp = utilities.transpose(cluster_train)
        temp = utilities.transpose(temp[:-1])
        # Obtain the clusters
        centroids, cluster_assignments = cluster.k_means(temp,clusters)
        # Determine the class of the cluster
        new_train  = cluster.get_new_train(dataset,len(dataset[0])-1,cluster_assignments,centroids)
        accuracy += nearest_neighbor.kNN_Test(dataset, new_train, test,len(dataset[0])-1,k3,kmeans=True)
    print("Percentage of total correct:",accuracy/5.0,"\n")
    print("Test complete","\n")
