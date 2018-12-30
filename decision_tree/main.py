from decision_tree import DecisionTree
from util import calculate_accuracy
import numpy

import random
import time
from sklearn.datasets import fetch_openml

random.seed(26)


def main():
    # bank marketing dataset
    bank_marketing = fetch_openml(data_id=1461)

    # only using 5k of the available 45211 instances
    data = bank_marketing.data[:5000]
    labels = bank_marketing.target[:5000]

    # converting nominal features to strings
    nominal_features = {1, 2, 3, 4, 6, 7, 8, 10, 15}

    converted_data = []
    num_rows = data.shape[0]
    num_cols = data.shape[1]

    for row in range(num_rows):
        new_row = []
        for col in range(num_cols):
            if col in nominal_features:
                new_row.append(str(data[row, col]))
            else:
                new_row.append(data[row, col])
        new_row.append(labels[row])
        converted_data.append(new_row)

    train_n = int(0.8 * num_rows)

    random.shuffle(converted_data)

    train = converted_data[:train_n]
    test = converted_data[train_n:]
    print("Number of training rows: {}".format(train_n))
    print("Number of testing rows: {}".format(len(test)))

    train_X = [row[:-1] for row in train]
    train_y = [row[-1] for row in train]
    test_X = [row[:-1] for row in test]
    test_y = [row[-1] for row in test]

    dt = DecisionTree()

    t1 = time.time()
    dt.fit(train_X, train_y)
    t2 = time.time()
    diff = t2 - t1
    print("Time to fit the decision tree: {:.2f} seconds".format(diff))
    predictions = [dt.predict(record) for record in test_X]

    accuracy = calculate_accuracy(predictions, test_y)
    print("Accuracy on the test set: {:.2f}%".format(accuracy * 100))


if __name__ == "__main__":
    main()