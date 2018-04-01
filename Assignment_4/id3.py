import utilities
import node
import copy

'''
This class performs the ID3 algorithm
'''


class ID3:
    def __init__(self, dataset, feature_indices):
        '''
        Constructor for the ID3 algorithm. Takes in the
        dataset and feature indices
        '''
        self.dataset = dataset
        self.feature_indices = feature_indices
        self.total_features = len(dataset[0]) - 1

    def __calculate_entropy(self, class_values, total):
        '''
        This method takes in the values for each classification
        and computes the entropy
        '''
        entropy = 0.0
        # Iterate through each class value
        for key in class_values:
            entropy += utilities.entropy(class_values[key]/total)
        return entropy

    def __get_feature_classes(self, features, classification):
        '''
        This method takes in the features and classifications and
        creates a dictionary
        '''
        feature_values = {}
        # Iterate through each feature entry
        for entry in range(len(features)):
            # Create a dictionary and set a counter if the
            # entry is not in the main dictionary
            if features[entry] not in feature_values:
                feature_values[features[entry]] = {}
                feature_values[features[entry]]['count'] = 0
            # Create a new dictionary entry if the classification value
            # is not associated with the feature entry
            if classification[entry] not in feature_values[features[entry]]:
                feature_values[features[entry]][classification[entry]] = 1
            # Increment the counter to track the occurences
            else:
                feature_values[features[entry]][classification[entry]] += 1
            # Increment the counter
            feature_values[features[entry]]['count'] += 1
        return feature_values

    def __calculate_gain(self, feature_dict, total_features, data_entropy):
        '''
        This method calculates the information gain of the feature
        '''
        # Create a dictionary that has the entropies for each feature
        entropies = {}
        # Iterate through each feature key
        for feature_key in feature_dict:
            entropy = 0.0
            count = feature_dict[feature_key]['count']
            # Iterate through each classification and calculate the entropy
            # for each key
            for class_key in feature_dict[feature_key]:
                if class_key != 'count':
                    entropy += utilities.entropy(feature_dict[feature_key][class_key]/count)
            # Calculate the proportional entropy
            entropies[feature_key] = (count/total_features) * entropy
        # Convert each entropy value to a negative
        negatives = [-x for x in list(entropies.values())]
        # Caulculate the gain of the feature
        return data_entropy + sum(negatives)

    def __categorical_gain(self, feature_index, data_entropy):
        '''
        This method calculates the gain for categorical features
        '''
        print("Determining the gain for categorical values")
        data = utilities.transpose(self.dataset)
        features = data[feature_index]
        total_features = len(features)
        # Create a dictory
        feature_values = self.__get_feature_classes(data[feature_index], data[-1])
        # Calculate the Gain
        return self.__calculate_gain(feature_values, total_features, data_entropy)

    def __continuous_gain(self, feature_index, data_entropy):
        '''
        This method calculates the information gain for continous features.
        The midpoint is taken between two entries and the gain is calculated,
        this is done for the entire list and the best gain with its threshold
        is returned
        '''
        print("Finding the best gain point of the continous values")
        data = utilities.transpose(self.dataset)
        total_features = len(data[feature_index])
        feature_class = list(zip(data[feature_index], data[-1]))
        feature_class.sort(key=lambda tup: tup[0])
        gains = []
        right_vals = {'count': 0}
        left_vals = {'count': 0}
        # Iterate through the sorted feature set and
        # build the right_vals dictionary
        for entry in feature_class:
            # Determine if the class is in the right_vals
            # partition. If it is increment the occurences
            # otherwise create the entry
            if entry[1] not in right_vals:
                right_vals[entry[1]] = 1
            else:
                right_vals[entry[1]] += 1
            # Increment the counter
            right_vals['count'] += 1
        # Iterate through the sorted feature set and move
        # the class of the entry to the left_vals dictionary,
        # remove it from the right_vals dictionary, and compute
        # the gain
        for entry in feature_class:
            # Move the class of the feature
            if entry[1] not in left_vals:
                left_vals[entry[1]] = 1
            else:
                left_vals[entry[1]] += 1
            left_vals['count'] += 1
            # Remove the class from right_vals and do
            # some bookkeeping
            right_vals[entry[1]] -= 1
            right_vals['count'] -= 1
            if right_vals[entry[1]] == 0:
                del right_vals[entry[1]]
            # Calculate the gain and add it to the gain list
            temp = {0: left_vals, 1: right_vals}
            gains.append(self.__calculate_gain(temp, total_features, data_entropy))
        print("Calculated a total of", len(gains), "gains")
        # Obtain the maximum gain and compute the threshold
        max_gain = max(gains)
        index = gains.index(max_gain)
        threshold = (feature_class[index][0] + feature_class[index+1][0])/2.0
        print("The maximum gain is", max_gain, "and the decision threshold is", threshold)
        return max_gain, threshold

    def __feature_entropy(self, feature_index, data_entropy):
        '''
        This method determines which calculation to use based on whether
        the feature is continuous or categorical
        '''
        threshold = None
        gain = 0.0
        # Use continous gain if the index is a float
        if isinstance(self.dataset[0][feature_index], float):
            print("Feature", feature_index, "contains continuous values")
            gain, threshold = self.__continuous_gain(feature_index, data_entropy)
        # Otherwise use categorical gain
        else:
            print("Feature", feature_index, "contains categorical values")
            gain = self.__categorical_gain(feature_index, data_entropy)
        return gain, threshold

    def __create_partitions(self, feature_index, gain, threshold):
        '''
        This method creates a new dataset based on the feature index_list
        and the threshold
        '''
        print("Creating a new data set based on feature", feature_index)
        partitions = {}
        # If threhold is None can assume that this is a categorical feature
        if threshold is None:
            # Iterate though each entry
            for entry in self.dataset:
                if entry[feature_index] not in partitions:
                    partitions[entry[feature_index]] = []
                key = entry[feature_index]
                del entry[feature_index]
                partitions[key].append(entry)
        # Otherwise it is a continous feature
        else:
            partitions = {"left": [], "right": []}
            # Iterate though each entry
            for entry in self.dataset:
                # If the value is less than or equal to the
                # threhold then add it to the left
                if entry[feature_index] <= threshold:
                    # Remove the value from the list
                    del entry[feature_index]
                    partitions["left"].append(entry)
                # Otherwuse add it to the right
                else:
                    # Remove the value from the list
                    del entry[feature_index]
                    partitions["right"].append(entry)
        return partitions

    def __determine_label(self, class_values):
        '''
        This method determines which class label has the highest count
        '''
        return max(class_values.items(), key=lambda x: x[1])[0]

    def build_tree(self):
        '''
        This method starts the ID3 algorithm
        '''
        # Create a Node
        root = node.Node()
        class_values = {}
        # Obtain all the classifications and their total
        # occurences.
        for entry in self.dataset:
            if entry[-1] not in class_values:
                class_values[entry[-1]] = 1
            else:
                class_values[entry[-1]] += 1
        # When the class_values is length of 1 then
        # there is a good chance that the dataset just
        # has 1 classification value.
        if len(class_values) == 1:
            for key in class_values:
                root.set_label(key)
                print("Created a leaf node with label", root.get_label())
                return root
        # When the classifications are only in the dataset
        # pick the highest occuring classification in the
        # dataset
        if len(self.dataset[0]) == 1:
            root.set_label(self.__determine_label(class_values))
            print("Created a leaf node with label", root.get_label())
            return root
        # Obtain the entropy over the whole data set
        data_entropy = self.__calculate_entropy(class_values, len(self.dataset))
        # Calculate the gain of each feature
        gains = []
        for feature in range(self.total_features):
            gains.append(self.__feature_entropy(feature, data_entropy))
        # Find which feature is the best
        best_feature = gains.index(max(gains, key=lambda x: x[0]))
        print("Determined the best feature to be", best_feature)
        # Set the feature in the node
        root.set_feature_index(self.feature_indices.pop(best_feature))
        # Create a new set of partitions
        part = self.__create_partitions(best_feature, gains[best_feature][0], gains[best_feature][1])
        # Set the threshold for the decision node
        if gains[best_feature][1] is not None:
            root.set_threshold(gains[best_feature][1])
        # Iterate through each partition and build a tree
        for key in part:
            features = copy.deepcopy(self.feature_indices)
            # Only process the partition if there is data
            # associated with it
            if len(part[key]) != 0:
                print("Building a decision node")
                node_build = ID3(part[key], features)
                root.set_branch(key, node_build.build_tree())
        return root
