from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import cifar10

from random import randrange

#https://www.machinecurve.com/index.php/2020/02/09/how-
#to-build-a-convnet-for-cifar-10-and-cifar-100-classification-with-keras/

#https://datascience.stackexchange.com/questions/46819/in-cifar-10-dataset

# cd C:\Users\danro\Documents\Meus Projetos\Workspace\PonyGE2\src
# python ponyge.py --parameters conv.txt


def build_model(phenotype, num_classes, input_shape):

    model = Sequential()
    phenotype = phenotype.split()
    first = True

    print(phenotype)

    for i in range(len(phenotype)):
        if phenotype[i] == "conv2D" and first:
            first = False
            model.add(Conv2D(int(phenotype[i+1]), (3,3), input_shape=input_shape))

        elif phenotype[i] == "conv2D":
            model.add(Conv2D(int(phenotype[i+1]), (3,3)))

        elif phenotype[i] == "max_pool" and first:
            first = False
            model.add(MaxPooling2D(pool_size=(2, 2), input_shape=input_shape))

        elif phenotype[i] == "max_pool":
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    return model

class conv(base_ff):

    def __init__(self):
        super().__init__()
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = tf.keras.utils.normalize(x_train, axis=1)
        x_test = tf.keras.utils.normalize(x_test, axis=1)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
    
    def evaluate(self, ind, **kwargs):

        #accuracy = randrange(1000)/1000
        model = build_model(ind.phenotype, 10, (32,32,3))
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
        results = model.fit(self.x_train, self.y_train, batch_size=64, epochs=10, validation_split=0.1, verbose=1)
        score = model.evaluate(self.x_test, self.y_test, verbose=1)
        accuracy = score[1]
        print("ACCURACY: ",accuracy)
        return accuracy
