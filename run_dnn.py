from models.DNN.model import Network
from tqdm import tqdm
import pandas as pd
import numpy as np


def get_data():
    dataset = pd.read_csv('data/train.csv')
    y_train = dataset['label']
    y_train = y_train.to_numpy()
    y_train = y_train.reshape(-1, 1)
    x_train = dataset.drop(
        ['label'], 
        axis=1
    )
    x_train = x_train.to_numpy()
    x_train = x_train.reshape(-1, 784, 1)
    x_train = x_train/256.0
    return x_train, y_train


def calc_accuracy(x_train, y_train):
    correct = 0
    for i, x_item in tqdm(enumerate(x_train)):
        pred = np.argmax(model.forward(x_item))
        if pred == y_train[i]:
            correct += 1
    return correct/len(y_train)




if __name__ == '__main__':
    x_train, y_train = get_data()
    model = Network()
    for epoch in range(25):
        print(f'Epoch: {epoch}')
        for i, x_item in tqdm(enumerate(x_train)):
            model.backward(x_item, y_train[i])
        accuracy = calc_accuracy(x_train, y_train)
        print(f'Accuracy: {accuracy}')

