import os
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D

import tensorflow_hub as hub

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping

from Data.statics import CLASS_NAMES_W_EN, CLASS_NAMES_W

inception_v3_path = "pretrained_models/inception_v3"
inception_resnet_v2_path = "pretrained_models\\inception_resnet_v2"
resnet50_path = "pretrained_models\\resnet_v2_50"
efficientNet_b6_path = "pretrained_models/EfficientNet_b6"
efficientNet_b0_path = "pretrained_models/EfficientNet_b0"
mobileNet_v2_path = "pretrained_models\\mobilenet_v2"
train_ds_path = "Data/images/train_img528"
val_ds_path = "Data/images/val_img528"
test_ds_path = "Data/images/test_img528"

# config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
# keras.backend.backend

batch_size = 32
img_height = 528
img_width = 528
n_class = 12

image_feature_description = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}


def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)


def read_dataset(batch):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_ds_path,
        image_size=(img_height, img_width),
        batch_size=batch)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_ds_path,
        image_size=(img_height, img_width),
        batch_size=batch)
    class_names = train_ds.class_names
    print(class_names)
    print("num of classes: {}".format(len(class_names)))
    return train_ds, val_ds


def test(model):
    # show_examples(model)
    classes = CLASS_NAMES_W_EN
    correct_num = 0
    imgs = []
    for index, name in enumerate(classes):
        class_path = test_ds_path + "\\" + name + '\\'
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = keras.preprocessing.image.load_img(
                img_path, target_size=(img_width, img_height)
            )
            img_array = keras.preprocessing.image.img_to_array(img)
            imgs.append(img)
            img_array = tf.expand_dims(img_array, 0)  # Create a batch

            predictions = model.predict(img_array)
            pre_label = np.argmax(predictions)
            print("pediction of {} is {}, while true label is {}".format(img_path, pre_label, index))
            if pre_label == index:
                correct_num += 1
    print("test accuracy is {}".format(correct_num/len(imgs)))


def show_examples(model):
    def plot_image(i, predictions_array, true_label, img):
        # predictions_array, true_label, img = predictions_array, true_label, img
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        class_names = CLASS_NAMES_W_EN

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                             100 * np.max(predictions_array),
                                             class_names[true_label]),
                   fontproperties='SimHei', color=color, fontsize=8)

    def plot_value_array(i, predictions_array, true_label):
        predictions_array, true_label = predictions_array, true_label
        plt.grid(False)
        plt.xticks(range(12))
        plt.yticks([])
        thisplot = plt.bar(range(12), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')

    # model = keras.Sequential([model, keras.layers.Softmax()])
    score = []
    imgs = []
    test_imgs = ["battery\\battery936.jpg", "biological\\biological918.jpg", "brown-glass\\brown-glass594.jpg",
                 "cardboard\\cardboard866.jpg", "clothes\\clothes993.jpg", "green-glass\\green-glass68.jpg",
                 "metal\\metal87.jpg", "paper\\paper990.jpg", "plastic\\plastic89.jpg", "shoes\\shoes982.jpg",
                 "trash\\trash89.jpg", "white-glass\\white-glass749.jpg"]
    for test_img_path in test_imgs:
        img = keras.preprocessing.image.load_img(
            test_ds_path + "\\" + test_img_path, target_size=(img_width, img_height)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        imgs.append((img))
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = model.predict(img_array)
        score.append(tf.nn.softmax(predictions[0]))

    num_rows = 4
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, score[i], i, imgs[i])
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, score[i], i)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # load the data
    train_ds, val_ds = read_dataset(batch_size)

    # define the model
    data_augmentation = tf.keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_width, img_height, 3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )
    normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
    inception = hub.KerasLayer(efficientNet_b6_path, trainable=False)
    early_stop = EarlyStopping(monitor="val_loss", patience=8, verbose=8)
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(img_width, img_height, 3)),
            normalization_layer,
            data_augmentation,
            inception,
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(batch_size=batch_size),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(batch_size=batch_size),
            layers.Dense(n_class)
        ]
    )

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    # step = tf.Variable(1, name="global_step", trainable=False)
    optimizer = tf.optimizers.Adam()

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    model.summary()

    history = model.fit(
        train_ds, epochs=300  ,
        validation_data=val_ds,
        callbacks=[early_stop]
    )

    model.save("my_models\\efficientNet_b6_transfer")

    # model.predict(val_ds)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = history.epoch

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    test(model)
    show_examples(model)

    # accuracy = tf.metrics.Accuracy()
    # mean_loss = tf.metrics.Mean(name="loss")
    # @tf.function
    # def train_step(inputs, labels):
    #     with tf.GradientTape as tape:
    #         logit = model(inputs)
    #         loss_value = loss(labels, logit)
    #
    #     gradients = tape.gradient(loss_value, model.trainable_variables)
    #     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #     step.assign_add(1)
    #
    #     accuracy.update_state(labels, tf.argmax(logit, -1))
    #     return loss_value
    #
    # train = train_ds.batch(32).prefetch(1)
    # val = val_ds.batch(32).prefetch(1)

    # num_epochs = 30
    # for i in range(num_epochs):
    #
    #     for example in train:
    #         img, label = example["image"], example["label"]
    #         loss_value = train_step(img, label)
    #         mean_loss.update_state(loss_value)
    #         tf.print(
    #             step, " loss: ", mean_loss.result(), " accuracy ", accuracy.result()
    #         )
    #
    # tf.print("## VALIDATION - ", i)
    # accuracy.reset_states()
    # for example in val:
    #     img, label = example["image"], example["label"]
    #     logit = model(img)
    #     accuracy.update_state(label, tf.argmax(logit, -1))
    # tf.print("accuracy: ", accuracy.result())
    # accuracy.reset_states()