import numpy as np
import os
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist

LOG_DIR = os.path.join('./', 'log')

def model_path(basepath, filename):
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    return os.path.join(basepath, filename)

encoding_dim = 32  # 784/32 = factor 24.5

input_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)

# autoencoder Model
autoencoder = Model(input_img, decoded)

# encoder Model
encoder = Model(input_img, encoded)

# decoder Model
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# load data
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print("{:7}: {}".format('x_train', x_train.shape))
print("{:7}: {}".format('x_test', x_test.shape))

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

print("save weights to:", model_path(LOG_DIR, 'autoencoder.h5'))
autoencoder.save(model_path(LOG_DIR, 'autoencoder.h5'))   # HDF5 file, you have to pip3 install h5py if don't have it

encoded_img = encoder.predict(x_test)
decoded_img = decoder.predict(encoded_img)

