import keras.backend as K
import tensorflow as tf
from keras import initializers, layers

def compute_norm(inputs):
    return K.sqrt(K.sum(K.square(inputs), -1))

def get_the_input(input_shape):
    return input_shape[:-1]

def assert_for_length(inputs):
    assert len(inputs) == 2

def one_hot_encoding(x):
    return K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

def batch_flatten_the_input(inputs, mask):
    return K.batch_flatten(inputs * K.expand_dims(mask, -1))


def return_is_tuple(input_shape):
    return tuple([None, input_shape[0][1] * input_shape[0][2]])

def return_is_not_tuple(input_shape):
    return tuple([None, input_shape[1] * input_shape[2]])

def sum_of_square(vectors, axis):
    return K.sum(K.square(vectors), axis, keepdims=True)

def normalized_the_data(s_squared_norm):
    return s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())

def get_input(num_capsule, dim_capsule, routings, channels, kernel_initializer):
    return num_capsule, dim_capsule, routings, channels, initializers.get(kernel_initializer)


def squash(vectors, axis=-1):
    s_squared_norm = sum_of_square(vectors, axis)
    scale = normalized_the_data(s_squared_norm)
    return scale * vectors

def get_shape_dim(input_shape):
    return input_shape[1], input_shape[2]

def not_defined_properly(input_shape):
    assert len(input_shape) >= 3, "not defined properly"

def error_in_code(input_num_capsule, channels):
     assert int(input_num_capsule/channels)/(input_num_capsule/channels)==1, "error"


def add_weights_for_layer_1(num_capsule, channels, dim_capsule, input_dim_capsule):
    return [num_capsule, channels, dim_capsule, input_dim_capsule]

def add_bias(num_capsule, dim_capsule):
    return [num_capsule,dim_capsule]

def add_weights_second_time(num_capsule,input_num_capsule, dim_capsule):
    return [num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]

def expand_inputs(inputs):
    return K.expand_dims(inputs, 1)

def input_tile(inputs_expand, num_capsule):
    return K.tile(inputs_expand, [1, num_capsule, 1, 1])


def repeat_elements_value(W, input_num_capsule, channels):
    return K.repeat_elements(W,int(input_num_capsule/channels),1)

def map_the_input(W2, inputs_tiled):
    return K.map_fn(lambda x: K.batch_dot(x, W2, [2, 3]) , elems=inputs_tiled)

def get_the_bias(inputs_hat,num_capsule, input_num_capsule):
    return tf.zeros(shape=[K.shape(inputs_hat)[0], num_capsule, input_num_capsule])

def assert_the_function(routings):
    assert routings > 0, 'invalid input.'

def squash_the_input(inputs_hat, b, B):
    return squash(K.batch_dot(tf.nn.softmax(b, dim=1), inputs_hat, [2, 2])+ B)

def create_tuple(arr):
    return tuple(arr)
