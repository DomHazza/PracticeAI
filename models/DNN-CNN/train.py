from model import DeepModel, ConvModel
from data_manager import DataManager
from log_manager import MetricLog
from tqdm import tqdm
import tensorflow as tf
import mlflow



NUM_EPOCHS = 10
BATCH_SIZE = 100

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()



@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return [loss, predictions]


@tf.function
def val_step(images, labels):
    predictions = model(images, training=False)
    loss = loss_fn(labels, predictions)
    return [loss, predictions]




if __name__ == '__main__':
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    mlflow.set_experiment("DNN digits")

    model = DeepModel()
    data_manager = DataManager()
    metric_log = MetricLog()
    x_train, y_train = data_manager.get_train()
    x_valid, y_valid = data_manager.get_valid()
    train_length = len(x_train)

    print('\nTraining...')
    with mlflow.start_run(run_name='CNN'):
        for epoch in tqdm(range(NUM_EPOCHS)):
            metric_log.reset_state()

            for min_index in range(0, train_length, BATCH_SIZE):
                max_index = min(train_length, min_index+BATCH_SIZE)
                loss, predictions = train_step(
                    x_train[min_index:max_index], 
                    y_train[min_index:max_index]
                )
            metric_log.log_train_epoch(
                epoch,
                y_train[min_index:max_index],
                predictions,
                loss
            )
            
            loss, predictions = val_step(x_valid, y_valid)
            metric_log.log_val_epoch(
                epoch,
                x_valid,
                y_valid,
                predictions,
                loss
            )
