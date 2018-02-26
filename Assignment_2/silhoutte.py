import math
import utilities

'''
This method calculates the overall silhoutte coefficent
for the data set
'''
def performance(dataset, clusters):
    dist_list = [[0.0 for cluster in range(max(clusters)+1)] for j in range(len(dataset))]
    count_list = [[0.0 for cluster in range(max(clusters)+1)] for j in range(len(dataset))]
    sil_list = [[0.0 for i in range(2)] for j in range(len(dataset))]
    coefficients = [0.0 for i in range(len(dataset))]
    # Iterate through each entry in the data set and for each entry
    # iterate through the clusters assignments, calculate the distance,
    # and store the value in the dist_list.
    for entry in range(len(dataset)):
         for cluster in range(len(clusters)):
             dist_list[entry][clusters[cluster]] += utilities.distance(dataset[entry], dataset[cluster])
             count_list[entry][clusters[cluster]] += 1.0
    # Iterate through dist_list and calculate the average distances for each
    # point and determine which of the other clusters is closest to the data
    # point
    for entry in range(len(dist_list)):
        dist_list[entry] = [sumation/total if total else 0 for sumation,total in zip(dist_list[entry],count_list[entry])]
        cluster = clusters[entry]
        sil_list[entry][0] = dist_list[entry].pop(cluster)
        sil_list[entry][1] = min(dist_list[entry])
    # Calculate the silhoutte coefficent for each data point
    for entry in range(len(coefficients)):
        max_val = max(sil_list[entry])
        if max_val != 0.0:
            coefficients[entry] = (sil_list[entry][1] - sil_list[entry][0])/max(sil_list[entry])
        else:
            coefficients[entry] = 0.0
    # Return the average silhoutte coefficent
    return sum(coefficients)/len(coefficients)
