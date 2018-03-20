import utilities


class ID3:
    def __init__(self, dataset, is_continuous):
        self.dataset = dataset
        self.is_continuous = is_continuous

    def _calculate_entropy(self, class_values, total):
        entropy = 0.0
        for key in class_values:
            temp = utilities.entropy(class_values[key]/total)
            print(temp)
            entropy += temp
        print("final entropy", entropy)
        return entropy

    def _categorical_gain(self, feature_index, class_index, data_entropy):
        data = utilities.transpose(self.dataset)
        features = data[feature_index]
        classification = data[class_index]
        total_features = len(features)
        feature_values = {}
        # Iterate through each feature entry
        for entry in range(total_features):
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
        # Create a dictionary that has the entropies for each feature
        entropies = {}
        # Iterate through each feature key
        for feature_key in feature_values:
            entropy = 0.0
            count = feature_values[feature_key]['count']
            # Iterate through each classification and calculate the entropy
            # for each key
            for class_key in feature_values[feature_key]:
                if class_key != 'count':
                    entropy += utilities.entropy(feature_values[feature_key][class_key]/count)
            # Calculate the proportional entropy
            entropies[feature_key] = (count/total_features) * entropy
        # Convert each entropy value to a negative
        negatives = [-x for x in list(entropies.values())]
        # Caulculate the gain of the feature
        print(data_entropy + sum(negatives))

    def _feature_entropy(self, feature_index, class_index):
        data = utilities.transpose(self.dataset)
        features = data[feature_index]
        classification = data[class_index]

        if self.is_continuous[feature_index]:
            pass
        else:
            pass

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
        data_entropy = self._calculate_entropy(class_values, len(self.dataset))
        # Calculate the gain of each feature
        self._categorical_gain(0, 8, data_entropy)
