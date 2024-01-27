import keras.backend as K
import tensorflow as tf
from keras import initializers, layers
from math_functions import *

class Length(layers.Layer):
    def get_config(self):
        return super(Length, self).get_config()

    def call(self, inputs, **kwargs):
        return compute_norm(inputs)

    def compute_output_shape(self, input_shape):
        return get_the_input(input_shape)

    



class Mask(layers.Layer):
    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            assert_for_length(inputs)
            inputs, mask = inputs
        else: 
            x = compute_norm(inputs)
            mask = one_hot_encoding(x)

        masked = batch_flatten_the_input(inputs, mask)
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:
            return return_is_tuple(input_shape)
        else:
            return return_is_not_tuple(input_shape)

    def get_config(self):
        return super(Mask, self).get_config()




class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_capsule,channels, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule, self.dim_capsule, self.routings, self.channels, self.kernel_initializer = get_input(num_capsule, dim_capsule, routings, channels, kernel_initializer)

    def build(self, input_shape):
        not_defined_properly(input_shape)
        self.input_num_capsule, self.input_dim_capsule =  get_shape_dim(input_shape)
        
        if(self.channels!=0):
            error_in_code(self.input_num_capsule, self.channels)
            self.W = self.add_weight(shape=add_weights_for_layer_1(self.num_capsule, self.channels, self.dim_capsule, self.input_dim_capsule),
                                     initializer=self.kernel_initializer,
                                     name='W')
            
            self.B = self.add_weight(shape=add_bias(self.num_capsule, self.dim_capsule),
                                     initializer=self.kernel_initializer,
                                     name='B')
        else:
            self.W = self.add_weight(shape=add_weights_second_time(self.num_capsule,self.input_num_capsule, self.dim_capsule),
                                     initializer=self.kernel_initializer,
                                     name='W')
            self.B = self.add_weight(shape=add_bias(self.num_capsule, self.dim_capsule),
                                     initializer=self.kernel_initializer,
                                     name='B')
        self.built = True

      

    def call(self, inputs, training=None):
        inputs_expand = expand_inputs(inputs)
        
        inputs_tiled = input_tile(inputs_expand, self.num_capsule)
        
        if(self.channels!=0):
            W2 = repeat_elements_value(self.W, self.input_num_capsule, self.channels)
        else:
            W2 = self.W
            
        inputs_hat = map_the_input(W2, inputs_tiled)

        b = get_the_bias(inputs_hat,self.num_capsule, self.input_num_capsule)

        assert_the_function(self.routings)
        for i in range(self.routings):
            outputs = squash_the_input(inputs_hat, b, self.B)

            if i < self.routings - 1:
                b += K.batch_dot(outputs, inputs_hat, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return create_tuple([None, self.num_capsule, self.dim_capsule])

def calculate_filter_value(dim_capsule,n_channels):
    return dim_capsule*n_channels

def target_shape_get_it(dim_capsule):
    return [-1, dim_capsule]

def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    output = layers.Conv2D(filters=calculate_filter_value(dim_capsule,n_channels), kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d')(inputs)
    outputs = layers.Reshape(target_shape=target_shape_get_it(dim_capsule), name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)