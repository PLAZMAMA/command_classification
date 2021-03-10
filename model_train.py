import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
from sklearn.utils import shuffle

class Model:
    preprocessor = hub.KerasLayer("http://tfhub.dev/tensorflow/albert_en_preprocess/3")

    def __init__(self):
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        encoder_inputs = self.preprocessor(text_input)
        encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_base/2")
        outputs = encoder(encoder_inputs)
        net = tf.keras.layers.Dense(256)(outputs['pooled_output'])
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(2, activation='softmax', dtype=tf.float32)(net)
        self.model = tf.keras.Model(text_input, net)

    def train(self, train_data, train_labels, epochs=5):
        """trains the model using the data given"""
        encoder_inputs = self.preprocessor(train_data)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(train_data, train_labels, batch_size=16, epochs=epochs)

    def evaluate(self, test_data, test_labels):
        """evaluates the model given the test data and labels"""
        results = self.model.evaluate(test_data, test_labels, batch_size=16)
        print(f'Total Loss: {results[0]}, Total Accuracy: {results[1] * 100}%')

if __name__ == '__main__':

    #creating instance of the model class to be used later and a train_portion for later use 
    model = Model()
    train_portion = 0.9 #how much of the data will go into training(0-1)

    #fetching the data from the csv files
    notes_data = pd.read_csv('notes_data.csv')['notes commands'].to_numpy(dtype=np.str)
    calories_data = pd.read_csv('calories_data.csv', delimiter=',')['calories commands'].to_numpy(dtype=np.str)

    #labeling and combining the data together
    data = np.array(notes_data.tolist() + calories_data.tolist(), dtype=np.str)
    labels = np.array([[1, 0]] * notes_data.shape[0] + [[1, 0]] * calories_data.shape[0])

    #mixing/shuffling the data the data
    shuffled_data, shuffled_labels = shuffle(data, labels)

    #split the data into train and test data
    train_test_mark = int(len(shuffled_data) * train_portion) #the index where the test data start and the training data stops
    train_data = shuffled_data[:train_test_mark]
    train_labels = shuffled_labels[:train_test_mark]
    test_data = shuffled_data[train_test_mark + 1:]
    test_labels = shuffled_labels[train_test_mark + 1:]

    #training the model(only the output layer currently will be trained)
    model.train(train_data, train_labels)

    #evaulating the model
    model.evaluate(test_data, test_labels)