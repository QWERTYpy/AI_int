import keras  # Для создания нейронной сети
from keras.datasets import mnist  # Набор цифр
from keras.models import Sequential  # Простейший тип сети. Вся информация передается только последующему слою
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
import tensorflow as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt  # Для отображения
case = int(input('Обучить - 1, Проверить - 2 , Свое - 3 :'))
if case == 1:
    (X_train, y_train), (X_valid, y_valid) = mnist.load_data()
    # Вывод блока ознакомления
    # plt.figure(figsize=(5,5))
    # for k in range(12):
    #     plt.subplot(3, 4, k+1)
    #     plt.imshow(X_train[k], cmap='Greys')
    #     plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    # Вывод конкретного числа
    # plt.imshow(X_valid[0], cmap='Greys')
    # plt.show()
    # print(X_valid[0])

    # Преобразование двумерных изображений в одномерные массивы
    X_train = X_train.reshape(60000, 784).astype('float32')
    X_valid = X_valid.reshape(10000, 784).astype('float32')

    # Преобразование целочисленных значений пикселов в вещественные
    X_train /= 255
    X_valid /= 255

    # Преобразование целочисленных меток в прямой код
    n_classes = 10
    y_train = to_categorical(y_train, n_classes)
    y_valid = to_categorical(y_valid, n_classes)

    # каждый слой сети последовательно (sequential) передает информацию только следующему слою
    model = Sequential()
    model.add(Dense(64, activation='sigmoid', input_shape=(784,)))
    model.add(Dense(10, activation='softmax'))

    # Обучение
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=128, epochs=200, verbose=1, validation_data=(X_valid, y_valid))

    # Сохранение обученной сети
    model.save('ai_int.h5')
if case == 2:
    n = 4  # Номер элемента для проверки
    _, (X_valid, y_valid) = mnist.load_data()
    y_it = y_valid[n]
    plt.imshow(X_valid[n], cmap='Greys')
    plt.show()
    X_valid = X_valid.reshape(10000, 784).astype('float32')
    X_valid /= 255
    n_classes = 10
    y_valid = to_categorical(y_valid, n_classes)
    model = keras.models.load_model('ai_int.h5')
    print(X_valid[n:n+1], type(X_valid[n:n+1]))
    res = model.predict(X_valid[n:n+1]).tolist()
    res = res[0]
    print(f'{y_it} - Это цифра - {res.index(max(res))}')

if case == 3:
    model = keras.models.load_model('test.h5')
    for i in range(10):
        your_image = f'num/{i}.png'
        # grayscale_image = cv2.imread(your_image, 0)
        color_image = cv2.imread(your_image)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        n_image = np.divide(gray_image, 255.0)

        plt.imshow(n_image, cmap='Greys')
        plt.show()
        n_image = n_image.reshape(1, 784).astype('float32')
        print(n_image, type(n_image))

        res = model.predict([n_image]).tolist()
        res = res[0]
        print(f'Это цифра - {res.index(max(res))}')




