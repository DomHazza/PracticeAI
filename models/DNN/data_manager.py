import pandas as pd



class DataManager():
    def __init__(self, split_ratio = [0.7, 0.15, 0.15]):
        x_data, y_data = self.__get_data()
        x_train, x_valid, x_test = self.__split_data(
            split_ratio, x_data
        )
        y_train, y_valid, y_test = self.__split_data(
            split_ratio, y_data
        )
        self.__train = [x_train, y_train]
        self.__valid = [x_valid, y_valid]
        self.__test = [x_test, y_test]


    def get_train(self):
        return self.__train
    

    def get_valid(self):
        return self.__valid
    

    def get_test(self):
        return self.__test


    def __get_data(self):
        dataset = pd.read_csv('data/digits.csv')
        y_data = dataset['label']
        y_data = y_data.to_numpy()
        y_data = y_data.reshape(-1, 1)
        y_data = y_data.astype("float32")
        x_data = dataset.drop(
            ['label'], 
            axis=1
        )
        x_data = x_data.to_numpy()
        x_data = x_data.reshape(-1, 28, 28)
        x_data = x_data/256.0
        x_data = x_data.astype("float32")
        return [x_data, y_data]


    def __split_data(self, split_ratio, data):
        num_items = len(data)
        train_index = int(split_ratio[0]*num_items)
        valid_index = int(split_ratio[1]*num_items+train_index)
        return [
            data[:train_index],
            data[train_index:valid_index],
            data[valid_index:]
        ]
