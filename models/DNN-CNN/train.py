from model import DeepModel, ConvModel
from data_manager import DataManager
from tqdm import tqdm
import tensorflow as tf
import mlflow



NUM_EPOCHS = 10
BATCH_SIZE = 100


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')



@tf.function
def train_step(image, labels):
    with tf.GradientTape() as tape:
        predictions = model(image, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def val_step(image, labels):
    predictions = model(image, training=False)
    loss = loss_fn(labels, predictions)
    val_loss(loss)
    val_accuracy(labels, predictions)






if __name__ == '__main__':
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    mlflow.set_experiment("DNN digits")

    model = DeepModel()
    data_manager = DataManager()
    x_train, y_train = data_manager.get_train()
    x_valid, y_valid = data_manager.get_valid()
    train_length = len(x_train)

    print('\nTraining...')
    with mlflow.start_run(run_name='CNN'):
        for epoch in tqdm(range(NUM_EPOCHS)):
            train_loss.reset_state()
            train_accuracy.reset_state()
            val_loss.reset_state()
            val_accuracy.reset_state()

            for min_index in range(0, train_length, BATCH_SIZE):
                max_index = min(train_length, min_index+BATCH_SIZE)
                train_step(
                    x_train[min_index:max_index], 
                    y_train[min_index:max_index]
                )
            
            val_step(x_valid, y_valid)

            mlflow.log_metric("loss", train_loss.result(), step=epoch)
            mlflow.log_metric("accuracy", train_accuracy.result(), step=epoch)
            mlflow.log_metric("val_loss", val_loss.result(), step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy.result(), step=epoch)
            # mlflow.log_image(image, "image.png")
        


