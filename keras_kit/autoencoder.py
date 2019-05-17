import numpy as np
import os
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.datasets import mnist
import matplotlib.pyplot as plt

LOG_DIR = os.path.join('./', 'log')
# DATA_DIR = '../../Dataset/mnist.npz'

def model_path(basepath, filename):
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    return os.path.join(basepath, filename)


class AutoEncoder(object):
    def __init__(self, encoding_dim, in_dim):
        self.encoding_dim = encoding_dim  # 784/32 = factor 24.5
        self.in_dim = in_dim

    def build(self):
        input_img = Input(shape=(self.in_dim,))
        encoded = Dense(self.encoding_dim, activation='relu')(input_img)
        ## if Adding a sparsity constraint on the encoded representations
        # add a Dense layer with a L1 activity regularizer
        # encoded = Dense(encoding_dim, activation='relu',
        #         activity_regularizer=regularizers.l1(10e-5))(input_img)
        decoded = Dense(self.in_dim, activation='sigmoid')(encoded)

        # autoencoder Model
        autoencoder = Model(input_img, decoded)

        # encoder Model
        encoder = Model(input_img, encoded)

        # decoder Model
        encoded_input = Input(shape=(self.encoding_dim,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(encoded_input, decoder_layer(encoded_input))
        return autoencoder, encoder, decoder
    
    def train(self, model, train_data, test_data):
        model.compile(optimizer='adadelta', loss='binary_crossentropy')
        model.fit(train_data, train_data,
                        epochs=50,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(test_data, test_data))

        print("save weights to:", model_path(LOG_DIR, 'autoencoder.h5'))
        # save whole model not only weights
        model.save(model_path(LOG_DIR, 'autoencoder.h5'))   


def data_gen():
    # load data
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    print("{:7}: {}".format('x_train', x_train.shape))
    print("{:7}: {}".format('x_test', x_test.shape))
    return x_train, x_test


n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)

if __name__ == '__main__':
    train_data, test_data = data_gen()
    #########
    # train
    #########
    # model = AutoEncoder(32, 784)
    # autoencoder, encoder, decoder = model.build()

    #########
    # predict
    # from saved model
    #########
    ## method 1
    autoencoder = load_model(model_path(LOG_DIR, 'autoencoder.h5'))
    print(autoencoder.layers)
    print(autoencoder.inputs)
    print(autoencoder.outputs)
    decoded_img = autoencoder.predict(test_data)

    ## method 2
    # TODO
    # encoded_img = encoder.predict(x_test)
    # decoded_img = decoder.predict(encoded_img)

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
