import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import csv

training_size = 33600  # 42000
testing_size = 8400
k_neighbors = 20

def read_data_files():
    # training data
    c = open("train.csv", "r")
    training_labels = []
    training_data = np.zeros(shape=(training_size, 784))
    testing_labels = []
    testing_data = np.zeros(shape=(testing_size, 784))
    i = 0
    lines = csv.reader(c)
    headers = next(lines)
    for line in lines:
        if i < training_size:
            training_labels.append(int(line[0]))
            temp = line[1:]
            training_data[i] = np.array(temp)
        else:
            testing_labels.append(int(line[0]))
            temp = line[1:]
            testing_data[i-training_size] = np.array(temp)
        i += 1
    return training_labels, training_data, testing_labels, testing_data


training_labels, training_data, testing_labels, testing_data = read_data_files()


def calculate_k_nearest_neighbors(training_labels, training_data, testing_labels, testing_data):
    confusion_matrix = np.zeros(shape=(10, 10), dtype=int)
    total = 0
    acc = 0
    nearest = open("nearest_split.csv", "w", newline='')
    neighbors = csv.writer(nearest)
    for i in range(testing_size):
        print("test", i, ":")
        dist = np.sqrt(np.sum(np.square(testing_data[i] - training_data), axis=1))
        nearest_nums = np.argsort(dist)
        neighbors.writerow(nearest_nums.tolist())
        count = [0] * 10
        for j in range(k_neighbors):
            count[training_labels[nearest_nums[j]]] += 1
        guess_num = count.index(max(count))
        real_num = testing_labels[i]
        confusion_matrix[real_num][guess_num] += 1
        if real_num == 8:
            total += 1
            if guess_num == 8:
                acc += 1
        print("guess number: {},   real number: {}".format(guess_num, real_num))
        print("-----------------------------------------")
    print(acc/total)
    print(confusion_matrix)
    nearest.close()


calculate_k_nearest_neighbors(training_labels, training_data, testing_labels, testing_data)



