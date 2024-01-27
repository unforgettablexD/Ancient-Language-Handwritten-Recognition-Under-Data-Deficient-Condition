import keras
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from keras.layers import Dense, Reshape
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
from keras.utils.vis_utils import plot_model

from utils import combine_images, load_emnist_balanced
from PIL import Image, ImageFilter
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from snapshot import SnapshotCallbackBuilder
import os
import numpy as np
import tensorflow as tf
import os
import argparse
import generate_data
K.set_image_data_format('channels_last')

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)


def build_decoder(n_class, input_shape):
    decoder = models.Sequential(name='decoder')
    decoder.add(Dense(input_dim=16 * n_class, activation="relu", units=7 * 7 * 32))
    decoder.add(Reshape((7, 7, 32)))
    decoder.add(BatchNormalization(momentum=0.8))
    decoder.add(layers.Deconvolution2D(32, 3, 3, subsample=(1, 1), border_mode='same', activation="relu"))
    decoder.add(layers.Deconvolution2D(16, 3, 3, subsample=(2, 2), border_mode='same', activation="relu"))
    decoder.add(layers.Deconvolution2D(8, 3, 3, subsample=(2, 2), border_mode='same', activation="relu"))
    decoder.add(layers.Deconvolution2D(4, 3, 3, subsample=(1, 1), border_mode='same', activation="relu"))
    decoder.add(layers.Deconvolution2D(1, 3, 3, subsample=(1, 1), border_mode='same', activation="sigmoid"))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))
    return decoder

def CapsNet(input_shape, n_class, routings):
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu', name='conv1')(x)
    conv2 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', activation='relu', name='conv2')(conv1)
    conv3 = layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv3')(conv2)
    primarycaps = PrimaryCap(conv3, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, channels=32, name='digitcaps')(primarycaps)
    out_caps = Length(name='capsnet')(digitcaps)

    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])
    masked = Mask()(digitcaps)

    decoder = build_decoder(n_class, input_shape)
    
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    return train_model, eval_model


def margin_loss(y_true, y_pred):
    positive_margin = K.maximum(0., 0.9 - y_pred)
    negative_margin = K.maximum(0., y_pred - 0.1)

    positive_term = y_true * K.square(positive_margin)
    negative_term = 0.5 * (1 - y_true) * K.square(negative_margin)

    L = positive_term + negative_term
    return K.mean(K.sum(L, axis=1))


def train_generator(x, y, batch_size, shift_fraction=0.):
    train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                       height_shift_range=shift_fraction)
    generator = train_datagen.flow(x, y, batch_size=batch_size)
    while 1:
        x_batch, y_batch = generator.next()
        yield ([x_batch, y_batch], [y_batch, x_batch])

def train(model, data, args):
    (x_train, y_train), (x_test, y_test) = data

    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=False, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    train_gen = train_generator(x_train, y_train, args.batch_size, args.shift_fraction)

    model.fit_generator(generator=train_gen,
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        shuffle=True,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=snapshot.get_callbacks(log, model_prefix=model_prefix))

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    return model


def test(model, data, args):
    x_test, y_test = data
    batch_size_multiplier = 8

    y_pred, x_recon = model.predict(x_test, batch_size=args.batch_size * batch_size_multiplier)

    print('-' * 30 + 'Begin: test' + '-' * 30)
    test_accuracy = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / float(y_test.shape[0])
    print('Test acc:', test_accuracy)

    return test_accuracy
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=16000, type=int)
    parser.add_argument('--verbose', default=False, type=bool)
    parser.add_argument('--cnt', default=190, type=int)
    parser.add_argument('-n', '--num_cls', default=10, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--samples_to_generate', default=10, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lr_decay', default=0.9, type=float)
    parser.add_argument('--lam_recon', default=0.392, type=float)
    parser.add_argument('-r', '--routings', default=3, type=int)
    parser.add_argument('--shift_fraction', default=0.1, type=float)
    parser.add_argument('--save_dir', default='./emnist_bal_200')
    parser.add_argument('-dg', '--data_generate', action='store_true')
    parser.add_argument('-w', '--weights', default=None)
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    (x_train, y_train), (x_test, y_test) = load_emnist_balanced(args.cnt)

    model, eval_model = CapsNet(input_shape=x_train.shape[1:],
                                n_class=len(np.unique(np.argmax(y_train, 1))),
                                routings=args.routings)
    if args.verbose:
        model.summary()

    M = 3
    nb_epoch = T = args.epochs
    alpha_zero = 0.01
    model_prefix = 'Model_'
    snapshot = SnapshotCallbackBuilder(T, M, alpha_zero, args.save_dir)

    if args.weights is not None:
        model.load_weights(args.weights)
    if not args.data_generate:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
        test(model=eval_model, data=(x_test, y_test), args=args)
    else:
        if args.weights is None:
            print('No weights are provided. You need to train a model first.')
        else:
            data_generator = dataGeneration(model=eval_model, data=((x_train, y_train), (x_test, y_test)),
                                            args=args, samples_to_generate=args.samples_to_generate)
            
