import tensorflow as tf
import pandas as pd
import numpy as np




class EncoderModel(tf.keras.Model):
    def __init__(self, img_height=28, img_width=28, batch_size=100, latent_dim=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.shape = [img_height, img_width]
        
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(26, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(13, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(latent_dim)
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(
                tf.math.reduce_prod(self.shape).numpy(), 
                activation='sigmoid'
            ),
            tf.keras.layers.Reshape(self.shape)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded




# class VariationalModel(tf.keras.Model):
#     def __init__(self, latent_dim = 2):
#         super().__init__()
#         self.encoder = tf.keras.Sequential(
#             [
#                 tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
#                 tf.keras.layers.Conv2D(
#                     filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
#                 tf.keras.layers.Conv2D(
#                     filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
#                 tf.keras.layers.Flatten(),
#                 tf.keras.layers.Dense(latent_dim + latent_dim),
#             ]
#         )
#         self.decoder = tf.keras.Sequential(
#             [
#                 tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
#                 tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
#                 tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
#                 tf.keras.layers.Conv2DTranspose(
#                     filters=64, kernel_size=3, strides=2, padding='same',
#                     activation='relu'),
#                 tf.keras.layers.Conv2DTranspose(
#                     filters=32, kernel_size=3, strides=2, padding='same',
#                     activation='relu'),
#                 # No activation
#                 tf.keras.layers.Conv2DTranspose(
#                     filters=1, kernel_size=3, strides=1, padding='same'),
#             ]
#         )


#     @tf.function
#     def sample(self, eps=None):
#         if eps is None:
#             eps = tf.random.normal(shape=(100, self.latent_dim))
#         return self.decode(eps, apply_sigmoid=True)


#     def encode(self, x):
#         mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
#         return mean, logvar


#     def reparameterize(self, mean, logvar):
#         eps = tf.random.normal(shape=mean.shape)
#         return eps * tf.exp(logvar * .5) + mean


#     def decode(self, z, apply_sigmoid=False):
#         logits = self.decoder(z)
#         if apply_sigmoid:
#             probs = tf.sigmoid(logits)
#             return probs
#         return logits



