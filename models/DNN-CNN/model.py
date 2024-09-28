import tensorflow as tf
import pandas as pd
import numpy as np




class DeepModel(tf.keras.Model):
    def __init__(self, img_height=28, img_width=28, batch_size=100):
        super().__init__()
        # self.i1 = tf.keras.layers.Input(shape=(img_height, img_width), batch_size=batch_size)
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(10, activation='softmax')


    def call(self, x):
        # x = self.i1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x




class ConvModel(tf.keras.Model):
    def __init__(self, img_height=28, img_width=28, batch_size=100):
        super().__init__()
        # self.i1 = tf.keras.layers.Input(shape=(img_height, img_width), batch_size=batch_size)
        self.c1 = tf.keras.layers.Conv2D(26, 3, activation='relu')
        self.p1 = tf.keras.layers.MaxPooling2D()
        self.c2 = tf.keras.layers.Conv2D(13, 3, activation='relu')
        self.p2 = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(50, activation='relu')
        self.d2 = tf.keras.layers.Dense(10, activation='softmax')


    def call(self, x):
        # x = self.i1(x)
        x = self.c1(x)
        x = self.p1(x)
        x = self.c2(x)
        x = self.p2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x




