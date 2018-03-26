class Node:
    def __init__(self, label=None, feature_index=None):
        self.feature_index = feature_index
        self.branches = {}
        self.label = label

    def set_feature_index(self, feature_index):
        self.feature_index = feature_index

    def set_label(self, label):
        self.label = label

    def get_label(self):
        return self.label

    def set_branch(self, branch, node):
        self.branches[branch] = node
    
    def get_branches(self):
        return self.branches
