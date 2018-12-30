from util import compute_entropy, compute_information_gain, partition_classes, compute_purity
import numpy as np


class LeafNode(object):
    def __init__(self, prediction):
        self.prediction = prediction

    @property
    def prediction(self):
        return self._prediction

    @prediction.setter
    def prediction(self, value):
        self._prediction = value


class RootNode(object):
    def __init__(self, split_attribute=None, split_val=None, left_child=None, right_child=None):
        self.split_attribute = split_attribute
        self.split_val = split_val
        self.left_child = left_child
        self.right_child = right_child

    @property
    def split_attribute(self):
        return self._split_attribute

    @split_attribute.setter
    def split_attribute(self, value):
        self._split_attribute = value

    @property
    def left_child(self):
        return self._left_child

    @left_child.setter
    def left_child(self, node):
        self._left_child = node

    @property
    def right_child(self):
        return self._right_child

    @right_child.setter
    def right_child(self, node):
        self._right_child = node


class DecisionTree(object):
    def __init__(self, max_depth=10):
        self.tree = RootNode()
        self.used_attributes = set()
        self.max_depth = max_depth

    def fit(self, X, y):
        """
        Fits the decision tree recursively

        Args:
            X: list of list containing the features for each record
            y: list of class labels (0s and 1s)

        """
        self.used_attributes = set()

        def fit_recurs(X, y, depth=0):
            if len(np.unique(y)) == 1:
                return LeafNode(y[0])
            elif len(np.unique(X)) == 1 or len(X[0]) == len(self.used_attributes) or depth == self.max_depth:
                values, counts = np.unique(y, return_counts=True)
                return LeafNode(values[np.argmax(counts)])
            else:
                max_info_gain = -1
                best_split_attr = None
                best_split_val = None

                for i in range(len(X[0])):
                    if i in self.used_attributes:
                        continue
                    column_values = np.unique([row[i] for row in X])
                    for j in range(len(column_values)):
                        split_val = column_values[j]
                        X_left, X_right, y_left, y_right = partition_classes(X, y, i, split_val)
                        info_gain = compute_information_gain(y, [y_left, y_right])
                        if info_gain > max_info_gain:
                            max_info_gain = info_gain
                            best_split_attr = i
                            best_split_val = split_val

                if best_split_attr is not None:
                    values, counts = np.unique(y, return_counts=True)
                    X_left, X_right, y_left, y_right = partition_classes(X, y, best_split_attr, best_split_val)

                    if not X_left:
                        return LeafNode(values[np.argmax(counts)])
                    elif not X_right:
                        return LeafNode(values[np.argmax(counts)])
                    else:
                        self.used_attributes.add(best_split_attr)
                        depth = depth + 1
                        left_child = fit_recurs(X_left, y_left, depth)
                        right_child = fit_recurs(X_right, y_right, depth)
                        return RootNode(best_split_attr, best_split_val, left_child, right_child)

        self.tree = fit_recurs(X, y)

    def predict(self, record):
        """
        Recursively predicts the label for an input record

        Args:
            record: list containing the features to perform classification

        Returns:
            prediction as a 0 or 1
        """
        def predict_recurs(tree, record):
            if isinstance(tree, LeafNode):
                return tree.prediction
            else:
                if tree.split_attribute is str:
                    if tree.split_val == record[tree.split_attribute]:
                        return predict_recurs(tree.left_child, record)
                    else:
                        return predict_recurs(tree.right_child, record)
                else:
                    if record[tree.split_attribute] <= tree.split_val:
                        return predict_recurs(tree.left_child, record)
                    else:
                        return predict_recurs(tree.right_child, record)

        return predict_recurs(self.tree, record)
