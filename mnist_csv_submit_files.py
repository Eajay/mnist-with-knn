import numpy as np
import time

import csv

training_size = 42000  # 42000
testing_size = 28000   # 28000
k_neighbors = 100



def read_data_files():
    # training data
    c = open("train.csv", "r")
    training_labels = []
    training_data = np.zeros(shape=(training_size, 784))
    i = 0
    lines = csv.reader(c)
    headers = next(lines)
    for line in lines:
        training_labels.append(int(line[0]))
        temp = line[1:]
        training_data[i] = np.array(temp)
        i += 1

    # testing data
    c = open("test.csv", "r")
    lines = csv.reader(c)
    headers = next(lines)
    testing_data = np.zeros(shape=(testing_size, 784))
    i = 0
    for line in lines:
        testing_data[i] = np.array(line)
        i += 1

    return training_labels, training_data, testing_data


training_labels, training_data, testing_data = read_data_files()


def calculate_k_nearest_neighbors(training_labels, training_data, testing_data):
    distance = np.zeros(shape=(testing_size, training_size))
    confusion_matrix = np.zeros(shape=(10, 10), dtype=int)
    res = open("result.csv", "w", newline='')
    content = csv.writer(res)
    content.writerow(["ImageId", "Label"])

    nearest = open("nearestFile.csv", "w", newline='')
    neighbors = csv.writer(nearest)

    for i in range(testing_size):
        total_start = time.time()
        print("test", i, ":")
        start = time.time()
        dist = np.sqrt(np.sum(np.square(testing_data[i] - training_data), axis=1))
        end = time.time()
        print("calculate distance costs: ", end - start, "s")
        start = time.time()
        nearest_nums = np.argsort(dist)
        neighbors.writerow(nearest_nums.tolist())
        count = [0] * 10
        for j in range(k_neighbors):
            count[training_labels[nearest_nums[j]]] += 1
        guess_num = count.index(max(count))
        end = time.time()
        total_end = time.time()
        print("calculate k nearest neighbors costs: ", end - start, "s")
        print("the total cost in this test: ", total_end - total_start, "s")
        print("guess number: ", guess_num)
        print("-----------------------------------------")
        content.writerow([i+1, guess_num])

    res.close()
    nearest.close()

calculate_k_nearest_neighbors(training_labels, training_data, testing_data)



