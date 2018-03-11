import preprocess
import nearest_neighbor
import neural_network
import random
import utilities
import cluster
import time

'''
Perform the tests on the machine data set
'''
def do_machine():
    vendor_dict = {}
    model_dict = {}
    vendor_count = 0.0
    model_count = 0.0

    dataset = preprocess.read_file(filename="machine.data",split=',')

    for entry in range(len(dataset)):
        if dataset[entry][0] not in vendor_dict:
            vendor_dict[dataset[entry][0]] = vendor_count
            vendor_count += 0.1
        if dataset[entry][1] not in model_dict:
            model_dict[dataset[entry][1]] = vendor_count
            model_count += 0.1
        dataset[entry][0] = str(vendor_dict[dataset[entry][0]])
        dataset[entry][1] = str(model_dict[dataset[entry][1]])
        dataset[entry] = [float(s.replace(',','')) for s in dataset[entry]]
    # Normalize the data set
    dataset = utilities.normalize_data(dataset)
    # Create partisions
    partitions = preprocess.create_regress_partitions(5,dataset)
    # Variables for the algorithms
    accuracy = 0.0
    k1=1
    k2=5
    clusters = 25
    accuracy = 0.0
    print("Performing k-nearest neighbor with k =", k1)
    time.sleep(1)
    for key in partitions:
        train, test = utilities.get_train_test_sets(partitions,key)
        accuracy += nearest_neighbor.kNN_Test(dataset, train, test,len(dataset[0])-1,k1,classify=False)
    print("Mean squared error:",accuracy/5.0,"\n")
    print("Test complete","\n")
    accuracy = 0.0
    print("Performing k means clustering with",clusters,"clusters")
    print("with k-nearest neighbors with k =", k2)
    input("Press Enter to continue...")
    time.sleep(1)
    for key in partitions:
        cluster_train = []
        train, test = utilities.get_train_test_sets(partitions,key)
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
        # Compute the accuracy
        accuracy += nearest_neighbor.kNN_Test(dataset, new_train, test,len(dataset[0])-1,k2,classify=False,kmeans=True)
    print("Mean squared error:",accuracy/5.0,"\n")
    print("Test complete","\n")
