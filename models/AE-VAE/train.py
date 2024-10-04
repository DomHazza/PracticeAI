from model import EncoderModel
from data_manager import DataManager
from log_manager import MetricLog
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import mlflow



NUM_EPOCHS = 20
BATCH_SIZE = 100

loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(1e-4)



# def log_normal_pdf(sample, mean, logvar, raxis=1):
#     log2pi = tf.math.log(2. * np.pi)
#     return tf.reduce_sum(
#         -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
#         axis=raxis
#     )


# def compute_loss(model, x):
#     mean, logvar = model.encode(x)
#     z = model.reparameterize(mean, logvar)
#     x_logit = model.decode(z)
#     cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
#     logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
#     logpz = log_normal_pdf(z, 0., 0.)
#     logqz_x = log_normal_pdf(z, mean, logvar)
#     return -tf.reduce_mean(logpx_z + logpz - logqz_x)



# @tf.function
# def train_step(images):
#     with tf.GradientTape() as tape:
#         loss = compute_loss(model, images)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     return loss


# @tf.function
# def val_step(images):
#     mean, logvar = model.encode(images)
#     z = model.reparameterize(mean, logvar)
#     predictions = model.sample(z)
#     loss = compute_loss(model, images)
#     return [loss, predictions]



@tf.function
def train_step(images):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(images, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return [loss, predictions]


@tf.function
def val_step(images):
    predictions = model(images, training=False)
    loss = loss_fn(images, predictions)
    return [loss, predictions]




if __name__ == '__main__':
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    mlflow.set_experiment("AE digits")

    model = EncoderModel()
    data_manager = DataManager()
    metric_log = MetricLog()
    image_train, label_train = data_manager.get_train()
    image_valid, label_valid = data_manager.get_valid()
    train_length = len(image_train)
    

    print('\nTraining...')
    with mlflow.start_run(run_name='AE'):
        for epoch in tqdm(range(NUM_EPOCHS)):
            metric_log.reset_state()

            for min_index in range(0, train_length, BATCH_SIZE):
                max_index = min(train_length, min_index+BATCH_SIZE)
                loss, predictions = train_step(
                    image_train[min_index:max_index]
                )
            metric_log.log_train_epoch(
                epoch,
                loss
            )
            loss, predictions = val_step(image_valid)
            encoded_images = model.encoder(image_valid)
            metric_log.log_val_epoch(
                epoch,
                predictions,
                loss
            )
            metric_log.log_scatter(
                epoch,
                encoded_images,
                label_valid
            )
            
