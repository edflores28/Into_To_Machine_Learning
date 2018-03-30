class Node:
    def __init__(self):
        self.feature_index = None
        self.branches = {}
        self.label = None
        self.is_leaf = False
        self.threshold = None
        self.is_pruned = False

    def get_leaf(self):
        return self.is_leaf

    def get_pruned(self):
        return self.is_pruned

    def set_feature_index(self, feature_index):
        self.feature_index = feature_index

    def set_label(self, label):
        self.label = label
        self.is_leaf = True

    def set_leaf(self, leaf):
        self.is_leaf = leaf

    def set_pruned(self, pruned):
        self.is_pruned = pruned

    def set_threshold(self, threshold):
        self.threshold = threshold

    def get_threshold(self):
        return self.threshold

    def set_branch(self, branch, node):
        self.branches[branch] = node

    def get_label(self):
        return self.label

    def get_branches(self):
        return self.branches

    def get_feature_index(self):
        return self.feature_index
