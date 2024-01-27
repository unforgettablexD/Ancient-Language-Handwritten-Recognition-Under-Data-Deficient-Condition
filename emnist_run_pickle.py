from tensorflow.keras.models import load_model
import tensorflow as tf


def load_emnist_balanced(cnt):
    from scipy import io as spio
    import numpy as np
    emnist = spio.loadmat("emnist-digits.mat")

    classes = 10
    cnt = cnt

    x_test = emnist["dataset"][0][0][1][0][0][0]
    x_test = x_test.astype(np.float32)
    y_test = emnist["dataset"][0][0][1][0][0][1]
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1, order="A").astype('float32') / 255.
    y_test = tf.keras.utils.to_categorical(y_test.astype('float32'))

    return  (x_test, y_test)

print("Please wait.....")
(x_test, y_test) = load_emnist_balanced(20000)
model = tf.keras.models.load_model('emnist_digits_model.hdf5')
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

print(f'Accuracy: {accuracy * 100} %')