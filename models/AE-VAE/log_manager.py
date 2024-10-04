import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import mlflow
import io



class MetricLog():
    def __init__(self):
        self.__train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.__val_loss = tf.keras.metrics.Mean(name='val_loss')


    def reset_state(self):
        self.__train_loss.reset_state()
        self.__val_loss.reset_state()


    def log_train_epoch(self, epoch, loss):
        self.__train_loss(loss)
        mlflow.log_metric(
            "loss", 
            self.__train_loss.result(), 
            step=epoch
        )


    def log_val_epoch(self, epoch, predicted_images, loss):
        self.__val_loss(loss)
        mlflow.log_metric(
            "val_loss", 
            self.__val_loss.result(), 
            step=epoch
        )
        image_file = self.__build_image(predicted_images)
        str_epoch = self.__correct_epoch_num(epoch)
        mlflow.log_image(image_file, f"epoch_{str_epoch}.png")


    def log_scatter(self, epoch, encoded_images, image_labels):
        fig, ax_1 = plt.subplots(1, 1, figsize=(10, 4))
        ax_1.scatter(
            x = encoded_images[:, 0],
            y = encoded_images[:, 1],
            c = image_labels,
            label = image_labels,
            cmap='turbo'
        )
        ax_1.legend()
        ax_1.set_title('Encoded space')
        image_file = self.fig_to_image(fig)
        plt.close()
        str_epoch = self.__correct_epoch_num(epoch)
        mlflow.log_image(image_file, f"scatter_{str_epoch}.png")


    def __correct_epoch_num(self, epoch):
        new_epoch = epoch+1000
        str_epoch = str(new_epoch)
        return str_epoch[1:]


    def __build_image(self, predicted_images):
        fig, axes = plt.subplots(4, 10, figsize=(10, 4))
        for i, row in enumerate(axes):
            for j, ax in enumerate(row):
                index = i*2 + j
                ax.imshow(predicted_images[index], cmap=plt.cm.binary)
        plt.tight_layout()
        img = self.fig_to_image(fig)
        plt.close()
        return img
    

    def fig_to_image(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img