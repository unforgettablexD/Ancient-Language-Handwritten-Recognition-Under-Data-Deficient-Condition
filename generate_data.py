import os
import argparse
import numpy as np
from keras import backend as K
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models, optimizers
from keras.layers import Dense, Reshape
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.utils.vis_utils import plot_model
from keras import callbacks
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from PIL import Image, ImageFilter
import keras
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
import utils

class dataGeneration():
    def __init__(self, model, data, args, samples_to_generate=2):
        self.model = model
        self.data = data
        self.args = args
        self.samples_to_generate = samples_to_generate
        print("-" * 100)

        (x_train, y_train), (x_test, y_test), x_recon = self.remove_missclassifications()
        self.data = (x_train, y_train), (x_test, y_test)
        self.reconstructions = x_recon
        self.inst_parameter, self.global_position, self.masked_inst_parameter = self.get_inst_parameters()
        self.x_decoder_retrain, self.y_decoder_retrain = self.decoder_retraining_dataset()
        self.retrained_decoder = self.decoder_retraining()
        self.class_variance, self.class_max, self.class_min = self.get_limits()
        self.generated_images, self.generated_labels = self.generate_data()

    def save_output_image(self, samples, image_name):
        if not os.path.exists(args.save_dir + "/images"):
            os.makedirs(args.save_dir + "/images")
        img = combine_images(samples)
        img = img * 255
        Image.fromarray(img.astype(np.uint8)).save(args.save_dir + "/images/" + image_name + ".png")

    def remove_missclassifications(self):
        model = self.model
        data = self.data
        args = self.args
        (x_train, y_train), (x_test, y_test) = data
        y_pred, x_recon = model.predict(x_train, batch_size=1)
        acc = np.sum(np.argmax(y_pred, 1) == np.argmax(y_train, 1)) / y_train.shape[0]
        cmp = np.argmax(y_pred, 1) == np.argmax(y_train, 1)
        bin_cmp = np.where(cmp == 0)[0]
        x_train = np.delete(x_train, bin_cmp, axis=0)
        y_train = np.delete(y_train, bin_cmp, axis=0)
        x_recon = np.delete(x_recon, bin_cmp, axis=0)
        self.save_output_image(x_train[:100], "original training")
        self.save_output_image(x_recon[:100], "original reconstruction")
        return (x_train, y_train), (x_test, y_test), x_recon

    def get_inst_parameters(self):
        model = self.model
        data = self.data
        args = self.args
        (x_train, y_train), (x_test, y_test) = data
        if not os.path.exists(args.save_dir + "/check"):
            os.makedirs(args.save_dir + "/check")

        if not os.path.exists(args.save_dir + "/check/x_inst.npy"):
            get_digitcaps_output = K.function([model.layers[0].input], [model.get_layer("digitcaps").output])

            get_capsnet_output = K.function([model.layers[0].input], [model.get_layer("capsnet").output])

            if (x_train.shape[0] % args.num_cls == 0):
                lim = int(x_train.shape[0] / args.num_cls)
            else:
                lim = int(x_train.shape[0] / args.num_cls) + 1

            for t in range(0, lim):
                if (t == int(x_train.shape[0] / args.num_cls)):
                    mod = x_train.shape[0] % args.num_cls
                    digitcaps_output = get_digitcaps_output([x_train[t * args.num_cls:t * args.num_cls + mod]])[0]
                    capsnet_output = get_capsnet_output([x_train[t * args.num_cls:t * args.num_cls + mod]])[0]
                else:
                    digitcaps_output = get_digitcaps_output([x_train[t * args.num_cls:(t + 1) * args.num_cls]])[0]
                    capsnet_output = get_capsnet_output([x_train[t * args.num_cls:(t + 1) * args.num_cls]])[0]
                masked_inst = []
                inst = []
                where = []
                for j in range(0, digitcaps_output.shape[0]):
                    ind = capsnet_output[j].argmax()
                    inst.append(digitcaps_output[j][ind])
                    where.append(ind)
                    for z in range(0, args.num_cls):
                        print("check j" + str(j) + "  z:" + str(z) + "   args.num_cls: " + str(args.num_cls))

                        if (z == ind):
                            continue
                        else:
                            digitcaps_output[j][z] = digitcaps_output[j][z].fill(0.0)
                    masked_inst.append(digitcaps_output[j].flatten())
                masked_inst = np.asarray(masked_inst)
                masked_inst[np.isnan(masked_inst)] = 0
                inst = np.asarray(inst)
                where = np.asarray(where)
                if (t == 0):
                    x_inst = np.concatenate([inst])
                    pos = np.concatenate([where])
                    x_masked_inst = np.concatenate([masked_inst])
                else:
                    x_inst = np.concatenate([x_inst, inst])
                    pos = np.concatenate([pos, where])
                    x_masked_inst = np.concatenate([x_masked_inst, masked_inst])
            np.save(args.save_dir + "/check/x_inst", x_inst)
            np.save(args.save_dir + "/check/pos", pos)
            np.save(args.save_dir + "/check/x_masked_inst", x_masked_inst)
        else:
            x_inst = np.load(args.save_dir + "/check/x_inst.npy")
            pos = np.load(args.save_dir + "/check/pos.npy")
            x_masked_inst = np.load(args.save_dir + "/check/x_masked_inst.npy")
        return x_inst, pos, x_masked_inst

    def decoder_retraining_dataset(self):
        model = self.model
        data = self.data
        args = self.args
        x_recon = self.reconstructions
        (x_train, y_train), (x_test, y_test) = data

        check_dir = os.path.join(args.save_dir, "check")
        if not os.path.exists(check_dir):
            os.makedirs(check_dir)

        x_decoder_retrain, y_decoder_retrain = self.load_or_preprocess_retrain_data(x_recon, check_dir)

        return x_decoder_retrain, y_decoder_retrain

    def load_or_preprocess_retrain_data(self, x_recon, check_dir):
        x_decoder_retrain_file = os.path.join(check_dir, "x_decoder_retrain.npy")
        y_decoder_retrain_file = os.path.join(check_dir, "y_decoder_retrain.npy")

        if not os.path.exists(x_decoder_retrain_file):
            x_decoder_retrain, y_decoder_retrain = self.preprocess_retrain_data(x_recon, check_dir)
            np.save(x_decoder_retrain_file, x_decoder_retrain)
            np.save(y_decoder_retrain_file, y_decoder_retrain)
        else:
            x_decoder_retrain = np.load(x_decoder_retrain_file)
            y_decoder_retrain = np.load(y_decoder_retrain_file)

        return x_decoder_retrain, y_decoder_retrain

    def preprocess_retrain_data(self, x_recon, check_dir):
        x_recon_sharped = self.sharpen_reconstructions(x_recon)
        self.save_output_image(x_recon_sharped[:100], "sharpened reconstructions")

        x_decoder_retrain = self.masked_inst_parameter
        y_decoder_retrain = x_recon_sharped

        return x_decoder_retrain, y_decoder_retrain

    def sharpen_reconstructions(self, x_recon):
        x_recon_sharped = np.empty([0])
        for q in range(0, x_recon.shape[0]):
            save_img = Image.fromarray((x_recon[q] * 255).reshape(28, 28).astype(np.uint8))
            image_more_sharp = save_img.filter(ImageFilter.UnsharpMask(radius=1, percent=1000, threshold=1))
            img_arr = np.asarray(image_more_sharp)
            img_arr = img_arr.reshape(-1, 28, 28, 1).astype('float32') / 255.
            x_recon_sharped = np.concatenate([x_recon_sharped, img_arr]) if x_recon_sharped.size else img_arr
        return x_recon_sharped

    def decoder_retraining(self):
        model = self.model
        data = self.data
        args = self.args
        x_decoder_retrain, y_decoder_retrain = self.x_decoder_retrain, self.y_decoder_retrain

        retrained_decoder = self.build_retrained_decoder(model, args)

        if not os.path.exists(args.save_dir + "/retrained_decoder.h5"):
            self.train_retrained_decoder(retrained_decoder, x_decoder_retrain, y_decoder_retrain, args)
            retrained_decoder.save_weights(args.save_dir + '/retrained_decoder.h5')
        else:
            retrained_decoder.load_weights(args.save_dir + '/retrained_decoder.h5')

        retrained_reconstructions = self.predict_retrained_decoder(retrained_decoder, x_decoder_retrain, args)
        self.save_output_image(retrained_reconstructions[:100], "retrained reconstructions")

        return retrained_decoder

    def build_retrained_decoder(self, model, args):
        decoder = model.get_layer('decoder')
        decoder_in = layers.Input(shape=(16 * args.num_cls,))
        decoder_out = decoder(decoder_in)
        retrained_decoder = models.Model(decoder_in, decoder_out)
        if args.verbose:
            retrained_decoder.summary()
        retrained_decoder.compile(optimizer=optimizers.Adam(lr=args.lr), loss='mse', loss_weights=[1.0])
        return retrained_decoder

    def train_retrained_decoder(self, retrained_decoder, x_decoder_retrain, y_decoder_retrain, args):
        retrained_decoder.fit(x_decoder_retrain, y_decoder_retrain, batch_size=1, epochs=20)

    def predict_retrained_decoder(self, retrained_decoder, x_decoder_retrain, args):
        return retrained_decoder.predict(x_decoder_retrain, batch_size=1)

    def get_limits(self):
        args = self.args
        x_inst = self.inst_parameter
        pos = self.global_position

        glob_min, glob_max = self.calculate_global_limits(x_inst)

        check_dir = os.path.join(args.save_dir, "check")
        if not os.path.exists(check_dir):
            os.makedirs(check_dir)

        class_cov, class_min, class_max = self.load_or_calculate_class_limits(x_inst, pos, args.num_cls, check_dir)

        return class_cov, class_max, class_min

    def calculate_global_limits(self, x_inst):
        glob_min = np.amin(x_inst.transpose(), axis=1)
        glob_max = np.amax(x_inst.transpose(), axis=1)
        return glob_min, glob_max

    def load_or_calculate_class_limits(self, x_inst, pos, num_cls, check_dir):
        class_cov_file = os.path.join(check_dir, "class_cov.npy")
        class_min_file = os.path.join(check_dir, "class_min.npy")
        class_max_file = os.path.join(check_dir, "class_max.npy")

        if not os.path.exists(class_cov_file):
            class_cov, class_min, class_max = self.calculate_class_limits(x_inst, pos, num_cls)
            np.save(class_cov_file, class_cov)
            np.save(class_min_file, class_min)
            np.save(class_max_file, class_max)
        else:
            class_cov = np.load(class_cov_file)
            class_min = np.load(class_min_file)
            class_max = np.load(class_max_file)

        return class_cov, class_min, class_max

    def calculate_class_limits(self, x_inst, pos, num_cls):
        class_cov = np.empty([0])
        class_min = np.empty([0])
        class_max = np.empty([0])

        for cl in range(0, num_cls):
            tmp_glob = self.extract_class_data(x_inst, pos, cl)
            tmp_cov_max, tmp_min, tmp_max = self.calculate_class_statistics(tmp_glob)

            class_cov, class_min, class_max = self.append_to_class_limits(class_cov, class_min, class_max,
                                                                          tmp_cov_max, tmp_min, tmp_max)

        return class_cov, class_min, class_max

    def extract_class_data(self, x_inst, pos, cl):
        tmp_glob = []
        for it in range(0, x_inst.shape[0]):
            if pos[it] == cl:
                tmp_glob.append(x_inst[it])
        return np.asarray(tmp_glob)

    def calculate_class_statistics(self, tmp_glob):
        tmp_glob = tmp_glob.transpose()
        tmp_cov_max = np.flip(np.argsort(np.around(np.cov(tmp_glob), 5).diagonal()), axis=0)
        tmp_min = np.amin(tmp_glob, axis=1)
        tmp_max = np.amax(tmp_glob, axis=1)
        return tmp_cov_max, tmp_min, tmp_max

    def append_to_class_limits(self, class_cov, class_min, class_max, tmp_cov_max, tmp_min, tmp_max):
        if class_cov.size == 0:
            class_cov = np.vstack([tmp_cov_max])
            class_min = np.vstack([tmp_min])
            class_max = np.vstack([tmp_max])
        else:
            class_cov = np.vstack([class_cov, tmp_cov_max])
            class_min = np.vstack([class_min, tmp_min])
            class_max = np.vstack([class_max, tmp_max])

        return class_cov, class_min, class_max

    def generate_data(self):
        data = self.data
        args = self.args
        (x_train, y_train), (x_test, y_test) = data
        x_masked_inst = self.masked_inst_parameter
        pos = self.global_position
        retrained_decoder = self.retrained_decoder
        class_cov = self.class_variance
        class_max = self.class_max
        class_min = self.class_min
        samples_to_generate = self.samples_to_generate

        generated_images, generated_images_with_ori, generated_labels = self.generate_images(x_train, x_masked_inst,
                                                                                             pos,
                                                                                             retrained_decoder,
                                                                                             class_cov,
                                                                                             class_max, class_min,
                                                                                             samples_to_generate,
                                                                                             args.num_cls)

        self.save_output_image(generated_images, "generated_images")
        self.save_output_image(generated_images_with_ori, "generated_images with originals")

        generated_labels = keras.utils.to_categorical(generated_labels, num_classes=args.num_cls)
        generated_data_dir = args.save_dir + "/generated_data"
        if not os.path.exists(generated_data_dir):
            os.makedirs(generated_data_dir)

        np.save(os.path.join(generated_data_dir, "generated_images"), generated_images)
        np.save(os.path.join(generated_data_dir, "generated_label"), generated_labels)

        return generated_images, generated_labels

    def generate_images(self, x_train, x_masked_inst, pos, retrained_decoder, class_cov, class_max, class_min,
                        samples_to_generate, num_cls):
        generated_images = np.empty([0, x_train.shape[1], x_train.shape[2], x_train.shape[3]])
        generated_images_with_ori = np.empty([0, x_train.shape[1], x_train.shape[2], x_train.shape[3]])
        generated_labels = np.empty([0])

        for cl in range(0, num_cls):
            count = 0
            for it in range(0, x_masked_inst.shape[0]):
                if count == samples_to_generate:
                    break
                if pos[it] == cl:
                    count = count + 1
                    generated_images_with_ori = np.concatenate(
                        [generated_images_with_ori,
                         x_train[it].reshape(1, x_train.shape[1], x_train.shape[2], x_train.shape[3])])
                    noise_vec = x_masked_inst[it][x_masked_inst[it].nonzero()]
                    for inst in range(int(class_cov.shape[1] / 2)):
                        ind = np.where(class_cov[cl] == inst)[0][0]
                        noise = np.random.uniform(class_min[cl][ind], class_max[cl][ind])
                        noise_vec[ind] = noise
                    x_masked_inst[it][x_masked_inst[it].nonzero()] = noise_vec
                    new_image = retrained_decoder.predict(x_masked_inst[it].reshape(1, num_cls * class_cov.shape[1]))
                    generated_images = np.concatenate([generated_images, new_image])
                    generated_labels = np.concatenate([generated_labels, np.asarray([cl])])
                    generated_images_with_ori = np.concatenate([generated_images_with_ori, new_image])

        return generated_images, generated_images_with_ori, generated_labels