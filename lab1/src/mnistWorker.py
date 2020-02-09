import sys
from random import *

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.special
from keras.datasets import mnist


class MyNN:
    def __init__(self, rate, inputs, hiddens, outputs):
        # добавляем один вход под bias
        self.i_count = inputs + 1
        self.h_count = hiddens
        self.o_count = outputs
        # заполняем массивы весов случайными значениями
        self.w_ih = np.random.normal(0.0, pow(self.h_count, -0.5), (self.h_count, self.i_count))
        self.w_ho = np.random.normal(0.0, pow(self.o_count, -0.5), (self.o_count, self.h_count))
        # learning rate и сигмоид
        self.lr = rate
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        # добавляем 1 вход под bias
        inputs_list = np.concatenate((inputs_list, [1]), axis=0)
        # вектор-столбцы входных данных и правильных ответов
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # прямое распространение, сигмоид и линеар
        hid_results = self.activation_function(np.dot(self.w_ih, inputs))
        out_results = self.activation_function(np.dot(self.w_ho, hid_results))
        # ошибки вывода
        out_errors = (targets - out_results)
        # ошибки скрытого слоя
        hid_errors = np.dot(self.w_ho.T, out_errors)
        # поправки для весов скрытый-выход
        self.w_ho += self.lr * np.dot(out_errors * out_results * (1.0 - out_results),
                                      np.transpose(hid_results))
        # поправки для весов вход-скрытый
        self.w_ih += self.lr * np.dot(hid_errors * hid_results * (1.0 - hid_results),
                                      np.transpose(inputs))

    def query(self, inputs_list):
        # добавляем 1 вход под bias
        inputs_list = np.concatenate((inputs_list, [1]), axis=0)
        # вектор-столбец входных данных
        inputs = np.array(inputs_list, ndmin=2).T
        # прямое распространение, сигмоид и линеар
        hid_results = self.activation_function(np.dot(self.w_ih, inputs))
        out_results = self.activation_function(np.dot(self.w_ho, hid_results))
        return out_results

    def set_lr(self, rate):
        self.lr = rate

def train(n):
    target = np.zeros(10)
    target[y_train[n]] = 1
    query = np.array(x_train[n]/255).reshape(784)
    myNN.train(query, target)

def trainR(n):
    target = np.zeros(10)
    target[y_train[n]] = 1
    rotation = random()*30-15
    imageR = scipy.ndimage.rotate(x_train[n]/255, rotation, cval=0, reshape=False)
    query = np.array(imageR).reshape(784)
    myNN.train(query, target)

def test_t(n):
    query = np.array(x_train[n] / 255).reshape(784)
    return myNN.query(query)

def test(n):
    query = np.array(x_test[n]/255).reshape(784)
    return myNN.query(query)

def epoch_train(learning_rate):
    myNN.set_lr(learning_rate)
    x_train_len = len(x_train)
    for i in range(x_train_len):
        trainR(i)
        if i%100 == 0:
            sys.stdout.write("Row: %s\r" % i)
            sys.stdout.flush()

def epoch_test():
    x_test_len = len(x_test)
    precision = 0
    i = 0
    for i in range (x_test_len):
        ans = test(i)
        if ans.argmax() == y_test[i]:
            precision += 1
    return precision/(i+1)

def epoch_test_t():
    x_test_len = len(x_train)
    precision = 0
    i = 0
    for i in range (x_test_len):
        ans = test_t(i)
        if ans.argmax() == y_train[i]:
            precision += 1
    return precision/(i+1)

def epoch_test_draw():
    x_test_len = len(x_test)
    precision = 0
    i = 0
    for i in range(x_test_len):
        ans = test(i)
        if ans.argmax() == y_test[i]:
            precision += 1
        else:
            plt.imshow(255-x_test[i], cmap="gray")
            plt.show()
            plt.pause(0.1)
    return precision/(i+1)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    myNN = MyNN(0.1, 784, 100, 10)

    for j in range (7):

        print("\nЭпоха ", j)

        epoch_train(0.1)

        print("\nНа обучающей:", epoch_test_t())
        print("На тестовой:", epoch_test())

    for k in range (3):
        epoch_train(0.01)

        print("\nПосле уменьшения learning rate на порядок, эпоха: ", k)
        print("\nНа обучающей:", epoch_test_t())
        print("На тестовой:", epoch_test())

    # print("\nРисуем цифры, которые неверно классифицированы")
    # print(epoch_test_draw())

    # лучший результат на тестовой 0.9785


