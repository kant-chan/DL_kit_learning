# -*- coding: utf-8 -*-
import numpy as np
import os
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.datasets import mnist

LOG_DIR = os.path.join('./', 'log')

def model_path(basepath, filename):
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    return os.path.join(basepath, filename)


def data_gen():
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    return x_train, x_test


def add_noise(data, f):
    noise_factor = f  # example f = 0.5
    x_train_noisy = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    return x_train_noisy


##### add noise, net construct is: #####

# input_img = Input(shape=(28, 28, 1))
# x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
# encoded = MaxPooling2D((2, 2), padding='same')(x)
# # at this point the representation is (7, 7, 32)
# x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
# x = UpSampling2D((2, 2))(x)
# decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

#####

class ConvAutoencoder(object):
    
    def build(self):
        input_img = Input(shape=(28, 28, 1))

        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        # at this point the representation is (4, 4, 8) i.e. 128-dimensional
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        self.autoencoder_model = Model(input_img, decoded)

    def train(self, train_data, train_label, test_data, test_label):
        self.autoencoder_model.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.autoencoder_model.fit(train_data, train_label,
                                   epochs=50,
                                   batch_size=128,
                                   shuffle=True,
                                   validation_data=(test_data, test_label),
                                   callbacks=[TensorBoard(log_dir=LOG_DIR)])

        print("save weights to:", model_path(LOG_DIR, 'convolutional_autoencoder.h5'))
        self.autoencoder_model.save_weights(model_path(LOG_DIR, 'convolutional_autoencoder.h5'))
        print('training completed!')

    def inference(self, x_test):
        self.autoencoder_model.load_weights(model_path(LOG_DIR, 'convolutional_autoencoder.h5'))
        decoded_img = autoencoder.predict(test_data)
        # visualize results
        n = 5
        plt.figure(figsize=(10, 4))
        for i in range(n):
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(test_data[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_img[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
        plt.show()

if __name__ == '__main__':
    train_data, test_data = data_gen()
    model = ConvAutoencoder()
    model.build()
    model.train(train_data, train_data, test_data, test_data)
