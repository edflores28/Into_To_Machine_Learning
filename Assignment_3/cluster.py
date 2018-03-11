import random
import utilities
import copy
import math

MAX_PRINT = 3

'''
This method recalculates the cluster centers by obtaining
the average data point inside the cluster.
'''
def recalc_centers(data_set, assignments, points):
    temp = [[0.0 for i in range(len(points[0]))] for j in range(len(points))]
    # Add all the entries that are associated with each cluster list
    for entry in range(len(assignments)):
        temp[assignments[entry]] = [sum(x) for x in zip(temp[assignments[entry]], data_set[entry])]
    # For each cluster find the total occurances and divide the total.
    # This calculates the new cluster.
    for entry in range(len(temp)):
        total = assignments.count(entry)
        if total != 0:
            temp[entry] = [i/assignments.count(entry) for i in temp[entry]]
    return temp;

'''
The k means algorithm. This method starts off by obtaining
random points in the data set for starting values.
Each datapoint is added to it's cluster based on how close
it is to the cluster center. The cluster center is updated
at each iteration.
'''
def k_means(data_set, k):
    global MAX_PRINT
    print ("Starting k means clustering.")
    c_points = [data_set[random.randint(0,len(data_set)-1)] for i in range(k)]
    c_old_points = []
    c_assignments = [0 for i in range(len(data_set))]
    count = 0
    distances = [[] for i in range(k)]
    for index in range(k):
        distances[index] = [0 for entry in range(len(data_set))]
    print("Starting cluster centers are:",c_points[:MAX_PRINT])
    # Exit the loops when there is no more changes
    # to the cluster points
    while c_points != c_old_points:
        # Calculate each distance from the centroid points for
        # each value.
        for entry in range(len(data_set)):
            for k_distance in range(k):
                distances[k_distance][entry] = utilities.distance(c_points[k_distance], data_set[entry])
        # Find which cluster is closest to the data point and assign the
        # data point to it's cluster
        for entry in range(len(c_assignments)):
            temp = []
            for k_dist in range(len(distances)):
                temp.append(distances[k_dist][entry])
            c_assignments[entry] = temp.index(min(temp))
        # Calculate the new averages for the cluster points
        c_old_points = copy.deepcopy(c_points)
        c_points = recalc_centers(data_set,c_assignments,c_points)
        if count <= 1:
            print("New cluster centers:", c_points[:MAX_PRINT])
        count += 1
    print("Finished k means clustering")
    return c_points, c_assignments

'''
This method produces a new training set based on the cluster
informations
'''
def get_new_train(data, class_index, cluster_assignments, centroids, classify=True):
    clusters = {}
    final_classes = {}
    empty = [0.0 for i in range(len(centroids[0]))]
    # Iterate through the cluster assignments and put them
    # in a dictionary
    for entry in range(len(cluster_assignments)):
        if cluster_assignments[entry] not in clusters:
            clusters[cluster_assignments[entry]] = []
        clusters[cluster_assignments[entry]].append(entry)
    # Perform the following for classification
    if classify:
        # Iterate through the clusters and determine the total
        # count of the classes withn that cluster
        for key in clusters:
            classes = {}
            for entry in range(len(clusters[key])):
                if data[clusters[key][entry]][class_index] not in classes:
                    classes[data[clusters[key][entry]][class_index]] = 0
                classes[data[clusters[key][entry]][class_index]] += 1
            max_val = -math.inf
            max_class = 0
            # Find the class with the highest number of points
            for class_key in classes:
                if classes[class_key] > max_val:
                    max_val = classes[class_key]
                    max_class = class_key
            # Assign the key with the most dominant class
            final_classes[key] = max_class
    # Perform the following  for regression
    else:
        # Iterate through the cluster and compute the average values
        # of their class
        for key in clusters:
            value = 0.0
            for item in clusters[key]:
                value += data[item][class_index]
            final_classes[key] = value/len(clusters[key])
    # Iterate through the cluster centroids and assign their
    # classes
    for entry in range(len(centroids)):
        try:
            centroids[entry].append(final_classes[entry])
        except:
            pass
    # In the case were there is nothing for a centroid remove the clusters
    # since it is empty
    for entry in centroids:
        if empty in centroids:
            del centroids[centroids.index(empty)]
    return centroids
