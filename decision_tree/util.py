import numpy as np


def compute_entropy(class_y):
    """
    Calculates the entropy of the input class labels

    Args:
        class_y: list of class labels

    Returns:
        entropy of the provided list as a float
    """

    num_labels = len(class_y)
    if num_labels == 0:
        return 0

    values, counts = np.unique(class_y, return_counts=True)
    probabilities = counts / num_labels

    entropy = -(probabilities * np.log2(probabilities)).sum()

    return entropy


def partition_classes(X, y, split_attribute, split_val):
    """
    Partitions both input data and labels based on the provided split attribute

    Args:
        X: list of list containing each record and the values for each attribute
        y: list of class labels
        split_attribute: column index of the attribute to split on
        split_val:  either a numerical or categorical value to divide the split_attribute

    Returns:
        A tuple containing the partitioned records/features and labels
    """
    X_left = []
    X_right = []

    y_left = []
    y_right = []

    if type(split_val) is str:
        for i in range(0, len(X)):
            if X[i][split_attribute] == split_val:
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                X_right.append(X[i])
                y_right.append(y[i])
    else:
        for i in range(0, len(X)):
            if X[i][split_attribute] <= split_val:
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                X_right.append(X[i])
                y_right.append(y[i])

    return X_left, X_right, y_left, y_right


def compute_information_gain(previous_y, current_y):
    """
    Computes information gain with the following formula:

        IG = H - (H_L * P_L + H_R * P_R)

    where H is the entropy of the previous node, H_L and H_R are the
    entropy of proposed splits and P_L and P_R are the probabilities
    of a value going to that side of the split

    Args:
        previous_y: list of class labels (0s and 1s)
        current_y: list of lists containing class labels (0s and 1s)

    Returns:
        information gain as a float
    """

    n = len(previous_y)
    H = compute_entropy(previous_y)

    H_L = compute_entropy(current_y[0])
    H_R = compute_entropy(current_y[1])

    information_gain = H - (H_L * len(current_y[0])/n + H_R * len(current_y[1])/n)

    return information_gain


def calculate_accuracy(predictions, labels):
    """
    Computes the accuracy by comparing the predicted label and the actual label

    Args:
        predictions: list of 0s and 1s
        labels: list of 0s and 1s

    Returns:
        accuracy as a float
    """
    # Comparing predicted and true labels
    results = [prediction == label for prediction, label in zip(predictions, labels)]

    # Accuracy
    accuracy = float(results.count(True)) / float(len(results))

    return accuracy


def compute_purity(y):
    """

    Args:
        y: list of labels (0s and 1s)

    Returns:
        dominant percentage of label as float
    """
    n = len(y)
    values, counts = np.unique(y, return_counts=True)

    if len(values) == 1:
        purity = 1.0
    else:
        purity = max(counts[0]/n, counts[1]/n)

    return purity

