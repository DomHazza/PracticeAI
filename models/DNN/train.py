from data_manager import DataManager
from model import DeepModel
from tqdm import tqdm
import tensorflow as tf
import pandas as pd
import numpy as np




if __name__ == '__main__':
    NUM_EPOCHS = 5
    model = DeepModel()
    data_manager = DataManager()
    x_train, y_train = data_manager.get_train()
    x_valid, y_valid = data_manager.get_valid()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer='adam',
        loss=loss_fn,
        metrics=['accuracy']
    )
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_valid,  y_valid, verbose=2)


