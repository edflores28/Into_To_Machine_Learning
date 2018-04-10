class Model:
    def __init__(self, train, test, labels):
        '''
        Initialization method
        '''
        self.train = train
        self.test = test
        self.labels = {}
        for entry in labels:
            self.labels[entry] = [0]
            self.labels[entry] += [[0, 0] for i in range(len(train[0])-1)]

    def __conv_pct(self, table):
        '''
        This method takes in a table and converts
        it into percentages
        '''
        for i in range(1, len(table)):
            table[i][0] = table[i][0]/table[0]
            table[i][1] = table[i][1]/table[0]

    def __predict(self, row):
        '''
        This method takes a row and determines which values
        are mostly likley to occur
        '''
        prediction = {}
        # Set the initial values for each class label
        for key in self.labels.keys():
            prediction[key] = self.labels[key][0]
        # Calulculate the probability for each class
        for entry in range(len(row[:-1])):
            for key in prediction.keys():
                prediction[key] *= self.labels[key][entry+1][row[entry]]
        # Return the class with the largest value
        return max(prediction.items(), key=lambda x: x[1])[0]

    def train_model(self):
        '''
        This method trains the naive bayes model
        '''
        for row in self.train:
            # Get the class label
            label = row[-1]
            # Increment the occurance of the class label
            self.labels[label][0] += 1
            # Count each entry in the row
            for entry in range(len(row[:-1])):
                self.labels[label][entry+1][row[entry]] += 1
        # Convert the values into probabilities
        for key in self.labels.keys():
            self.__conv_pct(self.labels[key])
            self.labels[key][0] /= len(self.train)

    def test_model(self):
        '''
        This method makes predictions for the training set
        '''
        correct = 0
        incorrect = 0
        # Iterate through each row and make a prediction
        for row in self.test:
            pred = self.__predict(row)
            if pred == row[-1]:
                correct += 1
            else:
                incorrect += 1
        print("Naive Bayes Statistics:")
        print("Percentage matched:\t", (correct*100)/(correct+incorrect),"\n")
