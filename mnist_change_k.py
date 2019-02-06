
import numpy as np

import csv


training_size = 33600  # 42000
testing_size = 8400
k_neighbors = 100

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

confusion_matrix = np.zeros(shape=(10, 10), dtype=int)
c = open("nearest_split.csv", "r")
lines = csv.reader(c)
i = 0
total = 0
cnt = 0
for line in lines:
    count = [0] * 10
    for j in range(k_neighbors):
        count[training_labels[int(line[j])]] += 1
    guess_num = count.index(max(count))
    real_num = testing_labels[i]
    confusion_matrix[real_num][guess_num] += 1
    print("test", i, ":")
    print("real number:", real_num)
    print("guess number:", guess_num)
    if real_num == 8:
        total += 1
        if guess_num == 8:
            cnt += 1
    i += 1
print(cnt/total)
print(confusion_matrix)

c.close()