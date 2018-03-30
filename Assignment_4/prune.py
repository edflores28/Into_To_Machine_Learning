import tree_utils
import copy


class Prune:
    def __init__(self, root, prune_set):
        self.root = root
        self.prune_set = prune_set

    def __get_leaves(self, node, labels):
        branches = node.get_branches()
        for key in branches:
            if branches[key].get_leaf():
                label = branches[key].get_label()
                if label not in labels:
                    labels[label] = 1
                else:
                    labels[label] += 1
            else:
                self.__get_leaves(branches[key], labels)
        return labels

    def __determine_label(self, node):
        labels = {}
        labels = self.__get_leaves(node, labels)
        return max(labels.items(), key=lambda x: x[1])[0]

    def __get_pruned_trees(self, root):
        branches = root.get_branches()
        for key in branches:
            if branches[key].get_leaf() is False and branches[key].get_pruned() is False:
                label = self.__determine_label(branches[key])
                branches[key].set_label(label)
                branches[key].set_pruned(True)
                return

    def reduced_error_prune(self, root):
        performance = {}
        # Determine the accuracy of the original tree
        #accuracy = tree_utils.predict_accuracy(self.prune_set, self.root)
        # Add the tree to the dictionary
        #performance[accuracy] = self.root
        # Obtain the branches of the root node
        branches = root.get_branches()
        # Only iterate through the interior nodes

        self.__get_pruned_trees(root)
        branches = root.get_branches()
        print(branches['left'].get_pruned())
        return performance
