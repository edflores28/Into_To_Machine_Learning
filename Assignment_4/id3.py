import utilities
from collections import OrderedDict


class ID3:
    def __init__(self, dataset, is_continuous):
        self.dataset = dataset
        self.is_continuous = is_continuous
        self.total_features = len(dataset[0]) - 1
    def __calculate_entropy(self, class_values, total):
        entropy = 0.0
        for key in class_values:
            entropy += utilities.entropy(class_values[key]/total)
        print("final entropy", entropy)
        return entropy

    def __get_feature_classes(self, features, classification):
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
        data = utilities.transpose(self.dataset)
        features = data[feature_index]
        total_features = len(features)
        # Create a dictory
        feature_values = self.__get_feature_classes(data[feature_index], data[-1])
        # Calculate the Gain
        return self.__calculate_gain(feature_values, total_features, data_entropy)

    def __continuous_gain(self, feature_index, data_entropy):
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
        # Obtain the maximum gain and compute the threshold
        max_gain = max(gains)
        index = gains.index(max_gain)
        threshold = (feature_class[index][0] + feature_class[index+1][0])/2.0
        return max_gain, threshold

    def __feature_entropy(self, feature_index, data_entropy):
        threshold = None
        gain = 0.0
        if isinstance(self.dataset[0][feature_index], float):
            gain, threshold = self.__continuous_gain(feature_index, data_entropy)
        else:
            gain = self.__categorical_gain(feature_index, data_entropy)
        print(gain,threshold)
        return gain, threshold

    def __create_partitions(self, feature_index, gain, threshold):
        partitions = {}
        if threshold is None:
            for entry in self.dataset:
                if entry[feature_index] not in partitions:
                    partitions[entry[feature_index]] = []
                else:
                    del entry[feature_index]
                    partitions[entry[feature_index]].append(entry)
        else:
            partitions = {"left": [], "right": []}
            for entry in self.dataset:
                if entry[feature_index] <= threshold:
                    del entry[feature_index]
                    partitions["left"].append(entry)
                else:
                    del entry[feature_index]
                    partitions["right"].append(entry)
        return partitions

    def build_tree(self):
        class_values = {}
        # Obtain all the classifications and their total
        # occurences.
        for entry in self.dataset:
            if entry[-1] not in class_values:
                class_values[entry[-1]] = 1
            else:
                class_values[entry[-1]] += 1
        # Obtain the entropy over the whole data set
        data_entropy = self.__calculate_entropy(class_values, len(self.dataset))
        # Calculate the gain of each feature
        gains = []
        for feature in range(self.total_features):
            gains.append(self.__feature_entropy(feature, data_entropy))
        best_feature = gains.index(max(gains))
        print("::::",gains[best_feature][0],gains[best_feature][1])
        part = self.__create_partitions(best_feature,gains[best_feature][0],gains[best_feature][1])
        for key in part:
            print(key, part[key],"\n")
