import random
import math
import copy
import utilities
import pdb

# Constant variables
MAX_ROWS_PRINT = 15
PRINT_COUNTER = 0;

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
    print ("Starting k means clustering.")
    c_points = [data_set[random.randint(0,len(data_set)-1)] for i in range(k)]
    c_old_points = []
    c_assignments = [0 for i in range(len(data_set))]
    count = 0
    distances = [[] for i in range(k)]
    for index in range(k):
        distances[index] = [0 for entry in range(len(data_set))]
    print("Starting cluster centers are:")
    for entry in c_points:
        print(entry)
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
        if count <= 10:
            print("New cluster centers:")
            for entry in c_points:
                print(entry)
        count += 1
    print("Finished k means clustering")
    return c_assignments

'''
This method used coplete linkage to obtain the
max distance between clusters.
'''
def complete_link_dist(cluster_a, cluster_b, data_set):
    max_dist = -math.inf
    # If the clusters are the same reutn 0.0
    if cluster_a == cluster_b :
        return 0.0
    for entry in range(len(cluster_a)):
        for other in range(len(cluster_b)):
            distance = utilities.distance(data_set[cluster_a[entry]], data_set[cluster_b[other]])
            if distance > max_dist:
                max_dist = distance
    return max_dist

'''
The Hierarchical agglomerative clustering algorithm. This genetic_algorithm
starts off with each data point as it's own cluster. Each clusters distance
is calculated with single linkage and the clusters closest to each other
are combined. This is done until there are k clusters.
'''
def HAC(data_set, k):
    print("Starting Hierarchical agglomerative clustering")
    c_assignments = [[i] for i in range(len(data_set))]
    # Loop until there are k total clusters
    count = 0
    while len(c_assignments) > k:
        matrix =[[0.0 for i in range(len(c_assignments))] for j in range(len(c_assignments))]
        # Compute the distances for all the available clusters and build
        # a table containing all the distances. The table gets filled in
        # diagonally such as row 1 gets filled then column 1 gets filled
        # then row 2 then column 2, etc..
        for clusters in range(len(c_assignments)):
            # Calculate the distance
            distances = [complete_link_dist(c_assignments[clusters], c_assignments[j], data_set) for j in range(clusters, len(c_assignments))]
            # Slice the matrix array to include values that were previously
            # filled
            temp = matrix[clusters][:clusters]
            # Add the calculated distanced to the temp row
            temp = temp + distances
            # Update the table
            matrix[clusters] = temp
            # Get a column representation of the table
            matrix = utilities.transpose(matrix)
            # Update the column
            matrix[clusters] = temp
            # Get a row representation of the table.
            matrix = utilities.transpose(matrix)
        # Set the following parameters to default values
        min_distance = math.inf
        min_row = 0
        min_column = 0
        # Traverse the matrix and find the lowest value.
        for row in range(len(matrix)):
            # Sort the array since the.
            temp = sorted(matrix[row])
            zeros = temp.count(0.0)
            # Set the minimum to the first non 0.0 index
            minimum = temp[zeros]
            # Determine if it is the minimum and update
            # the row and column
            col = matrix[row].index(minimum)
            if minimum < min_distance and col != row:
                min_distance = minimum
                min_row = row
                min_column = col
        # Traverse the minimum colum and append all the values
        # to consolidate the clusters
        if count <= 10:
            print("Combining closest clusters",min_row, min_column)
        if min_row != min_column:
            for entry in range(len(c_assignments[min_column])):
                c_assignments[min_row].append(c_assignments[min_column][entry])
                # Delete the minimum column entry for the cluster assignments
            del c_assignments[min_column]
        matrix.clear();
        count += 1
    # Create a new cluster list that has just the cluster values
    # mapped to the entry in the overall dataset
    new_assignments = [0 for i in range(len(data_set))]
    for row in range(len(c_assignments)):
        for entry in range(len(c_assignments[row])):
            new_assignments[c_assignments[row][entry]] = row
    print("Finished Hierarchical agglomerative clustering")
    return new_assignments
