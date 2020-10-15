from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import cifar10

from random import randrange

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
        results = model.fit(self.x_train,self.y_train, batch_size=64, epochs=2, validation_split=0.1, verbose=0)
        score = model.evaluate(self.x_test, self.y_test, verbose=0)
        accuracy = score[1]
        return accuracy


        """
        #print("HOUSTON "+ind.phenotype+"   ",rmd)

    	# olhar exemplo sequence_match
    	# pelo que aparenta, o ponyge gera uma sequencia de "palavras" pela gramática,
    	# depois nós pegamos essa sequencia e aplicamos como quisermos.

    	total = 0
        matches = 0
        with torch.no_grad():
	        for batch in train_data:
	        	X, y = batch
	        	output = net(X.view(-1,28*28))
	        	for idx, i in enumerate(output):
			      if torch.argmax(i) == y[idx]:
			        matches += 1
			      total += 1

		accuracy = matches/total
		"""
		