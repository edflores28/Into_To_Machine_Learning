import tree_utils
import copy
import node

'''
This class performs reduced error pruning on the decision tree
'''


class Prune:
    def __init__(self, root, prune_set):
        '''
        Constructor for the pruning algorithm. Takes in the
        root node and the pruning set
        '''
        self.root = root
        self.prune_set = prune_set

    def __get_leaves(self, thenode, labels):
        '''
        This method obtains all the labels of the leaves
        given thenode
        '''
        # Obtain the branches and iterate through them
        branches = thenode.get_branches()
        for key in branches:
            # If the node is a leaf get the label and
            # add it to the labels lst
            if branches[key].get_leaf():
                label = branches[key].get_label()
                if label not in labels:
                    labels[label] = 1
                else:
                    labels[label] += 1
            # Otherwise do recursion
            else:
                self.__get_leaves(branches[key], labels)
        return labels

    def __determine_label(self, thenode):
        '''
        This method obtains the label of the node
        and returns the most occuring label
        '''
        labels = {}
        labels = self.__get_leaves(thenode, labels)
        return max(labels.items(), key=lambda x: x[1])[0]

    def __get_pruned_trees(self, is_root, thenode):
        '''
        This method iterates through the entire tree and
        creates a new tree at each node, with the node
        turned into a leaf with the most common label
        '''
        subtree_list = []
        # Get the branches at the root node
        if is_root:
            branches = self.root.get_branches()
        # Otherwise get the branches at thenode
        else:
            branches = thenode.get_branches()
        # Iterate through the branches
        for key in branches:
            # When the node is an interior node
            if branches[key].get_leaf() is False:
                # Create a copy of the node
                node_copy = branches[key]
                # Determine the best label of the node
                label = self.__determine_label(branches[key])
                print("This node has", label, "as the most common label")
                # Create a new node and set the label
                new_node = node.Node()
                new_node.set_label(label)
                # Replace the node with the new node, this
                # prunes the tree.
                branches[key] = new_node
                # Copy the whole tree and add it to the list
                subtree_list.append(copy.deepcopy(self.root))
                # Replace the node with it's original self
                branches[key] = node_copy
                # Continue traversing the tree
                subtree_list += self.__get_pruned_trees(False, branches[key])
        return subtree_list

    def reduced_error_prune(self):
        '''
        This method starts the reduced error pruning algorithm
        '''
        print("Starting the pruning algorithm")
        performance = []
        # Obtain all the trees
        sublist = self.__get_pruned_trees(self.root, True)
        # Iterate through each tree and determine the accuracy with
        # the pruning set.
        for entry in sublist:
            accuracy = tree_utils.predict_accuracy(self.prune_set, entry)
            performance.append(accuracy)
        print("Generated a total of", len(sublist), "trees and accuracies")
        # Return the maximum performance
        return max(performance)
