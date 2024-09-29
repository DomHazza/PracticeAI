import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import mlflow
import io



class MetricLog():
    def __init__(self):
        self.__train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.__train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.__val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.__val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')


    def reset_state(self):
        self.__train_loss.reset_state()
        self.__train_accuracy.reset_state()
        self.__val_loss.reset_state()
        self.__val_accuracy.reset_state()


    def log_train_epoch(self, epoch, labels, predictions, loss):
        self.__train_loss(loss)
        self.__train_accuracy(labels, predictions)
        mlflow.log_metric(
            "loss", 
            self.__train_loss.result(), 
            step=epoch
        )
        mlflow.log_metric(
            "accuracy", 
            self.__train_accuracy.result(), 
            step=epoch
        )


    def log_val_epoch(self, epoch, images, labels, predictions, loss):
        self.__val_loss(loss)
        self.__val_accuracy(labels, predictions)
        mlflow.log_metric(
            "val_loss", 
            self.__val_loss.result(), 
            step=epoch
        )
        mlflow.log_metric(
            "val_accuracy", 
            self.__val_accuracy.result(), 
            step=epoch
        )
        image_files = self.__build_images(
            images, 
            labels, 
            predictions
        )
        for i, image_file in enumerate(image_files):
            mlflow.log_image(image_file, f"Validation_{i}.png")


    def __build_images(self, images, labels, predictions):
        image_files = []
        for start_index in range(0, 26, 10):
            fig, axes = plt.subplots(2, 5, figsize=(10, 4))
            for i, row in enumerate(axes):
                for j, ax in enumerate(row):
                    index = start_index + i*2 + j
                    ax.imshow(images[index], cmap=plt.cm.binary)
                    prediction = tf.argmax(predictions[index])
                    label = labels[index]
                    ax.set_title(f'Num {label}, Guess {prediction}')
            plt.tight_layout()
            img = self.fig_to_image(fig)
            image_files.append(img)
            plt.close()
        return image_files
    

    def fig_to_image(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img