'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import cifar10
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, UpSampling2D, AtrousConvolution2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils
from keras import backend as K

batch_size = 256
nb_classes = 10
nb_epoch = 75

# input image dimensions
img_rows, img_cols = 32, 32
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(LeakyReLU(alpha=.003))
#model.add(UpSampling2D(size=pool_size))
model.add(AtrousConvolution2D(64, 3, 3, border_mode='same'))
model.add(LeakyReLU(alpha=.003))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(LeakyReLU(alpha=.003))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.35))

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(LeakyReLU(alpha=.003))
#model.add(UpSampling2D(size=pool_size))
model.add(AtrousConvolution2D(64, 3, 3, border_mode='same'))
model.add(LeakyReLU(alpha=.003))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(LeakyReLU(alpha=.003))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.35))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
#              optimizer='adadelta',
              optimizer='adam',
              metrics=['accuracy'])

# Set up TensorBoard
tb = TensorBoard(log_dir='./logs')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test), callbacks=[tb])

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print("Accuracy: %.2f%%" % (score[1]*100))