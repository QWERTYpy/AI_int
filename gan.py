# для ввода и вывода данных:
import numpy as np
import os
# для глубокого обучения:
import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Dropout
from keras.layers import BatchNormalization, Flatten
from keras.layers import Activation
from keras.layers import Reshape  # новое!
from keras.layers import Conv2DTranspose, UpSampling2D
# from keras.optimizers import RMSprop
import tensorflow as tf
# tf.keras.optimizers.RMSprob
# для создания графика:
import pandas as pd
from matplotlib import pyplot as plt
# %matplotlib inline
from keras.optimizer_v2 import rmsprop

# Загрузка данных Quick, Draw!
input_images = "apple.npy"
data = np.load(input_images)
print(data.shape)

data = data/255  # Значения пикселов делятся на 255, чтобы привести их в диапазон от 0 до 1
data = np.reshape(data, (data.shape[0], 28, 28, 1))  # из одномерных массивов с 784 пикселами в двумерные матрицы 28 × 28 пикселов
img_w, img_h = data.shape[1:3]
plt.imshow(data[4242, :, :, 0], cmap='Greys')
#plt.show()

def build_discriminator(depth=64, p=0.4):
    # Определение входов
    image = Input((img_w, img_h, 1))  # Входные изображения имеют размер 28 × 28 пикселов
    # Сверточные слои
    conv1 = Conv2D(depth*1, 5, strides=2, padding='same', activation='relu')(image) # Размер фильтра 5 × 5
    conv1 = Dropout(p)(conv1)  # прореживание на уровне 40%
    conv2 = Conv2D(depth*2, 5, strides=2, padding='same', activation='relu')(conv1)
    conv2 = Dropout(p)(conv2)
    conv3 = Conv2D(depth*4, 5, strides=2, padding='same', activation='relu')(conv2)
    conv3 = Dropout(p)(conv3)
    conv4 = Conv2D(depth*8, 5, strides=1, padding='same', activation='relu')(conv3)
    conv4 = Flatten()(Dropout(p)(conv4))  # преобразуем трехмерную карту активации, полученную на выходе
    # конечного сверточного слоя, в плоский массив

    # выходной слой
    prediction = Dense(1, activation='sigmoid')(conv4)  #  бинарной классификации, из единственного сигмоидного нейрона
    # Определение модели
    model = Model(inputs=image, outputs=prediction)
    return model

discriminator = build_discriminator()
discriminator.summary()

discriminator.compile(loss='binary_crossentropy', optimizer=rmsprop(lr=0.0008, decay=6e-8, clipvalue=1.0), metrics=['accuracy'])

"""
мы используем бинарную функцию стоимости на основе перекрестной энтропии, потому что дискриминатор является моделью бинарной классификации
RMSprop, представленный в главе 9, — это один из «необычных оптимизаторов», служащий альтернативой для Adam
Скорость затухания (decay, ρ) в оптимизаторе RMSprop является гиперпараметром
clipvalue — это гиперпараметр, не позволяющий градиенту обучения (отношению частной производной между стоимостью и 
значениями параметров во время стохастического градиентного спуска) превысить это значение; таким образом, 
clipvalue явно ограничивает взрыв градиента
"""
# Генератор
z_dimensions = 32


def build_generator(latent_dim=z_dimensions,
                    depth=64, p=0.4):
    # Define inputs
    noise = Input((latent_dim,))

    # First dense layer
    dense1 = Dense(7 * 7 * depth)(noise)
    dense1 = BatchNormalization(momentum=0.9)(dense1)  # default momentum for moving average is 0.99
    dense1 = Activation(activation='relu')(dense1)
    dense1 = Reshape((7, 7, depth))(dense1)
    dense1 = Dropout(p)(dense1)

    # De-Convolutional layers
    conv1 = UpSampling2D()(dense1)
    conv1 = Conv2DTranspose(int(depth / 2), kernel_size=5, padding='same', activation=None, )(conv1)  # Сеть содержит три развертывающих слоя
    conv1 = BatchNormalization(momentum=0.9)(conv1)
    conv1 = Activation(activation='relu')(conv1)

    conv2 = UpSampling2D()(conv1)
    conv2 = Conv2DTranspose(int(depth / 4),
                            kernel_size=5, padding='same',
                            activation=None, )(conv2)
    conv2 = BatchNormalization(momentum=0.9)(conv2)
    conv2 = Activation(activation='relu')(conv2)

    conv3 = Conv2DTranspose(int(depth / 8),
                            kernel_size=5, padding='same',
                            activation=None, )(conv2)
    conv3 = BatchNormalization(momentum=0.9)(conv3)
    conv3 = Activation(activation='relu')(conv3)

    # Output layer
    image = Conv2D(1, kernel_size=5, padding='same',
                   activation='sigmoid')(conv3)

    # Model definition
    model = Model(inputs=noise, outputs=image)

    return model

generator = build_generator()

generator.summary()
