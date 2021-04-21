import numpy as np
import tensorflow as tf
from tensorflow import keras

from class_names import CLASS_NAMES_W_EN

img_width = 299
img_height = 299


def sort(img_path: str, model):
    classes = CLASS_NAMES_W_EN
    correct_num = 0
    img = keras.preprocessing.image.load_img(
        img_path, target_size=(img_width, img_height)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    pre_label = np.argmax(predictions)
    # print("pediction of {} is {}".format(img_path, pre_label))
    x = tf.nn.softmax(predictions[0])
    x = x.numpy()
    # print(x)
    return x


if __name__ == '__main__':
    model = keras.models.load_model('myModel/inceptionv3-transfer-94')
    sort("static/upload/img/-7b41ad00a1e84250.jpg", model)