import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
from sklearn.utils import shuffle

# def build_model(max_len=512):
#     """build the model"""
#     #setting up the input layer
#     input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
#     input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
#     segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

#     #fetching the beat model
#     bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_base/2")

#     #
#     pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
#     clf_output = sequence_output[:, 0, :]
#     net = tf.keras.layers.Dense(64, activation='relu')(clf_output)
#     net = tf.keras.layers.Dropout(0.2)(net)
#     net = tf.keras.layers.Dense(32, activation='relu')(net)
#     net = tf.keras.layers.Dropout(0.2)(net)
#     out = tf.keras.layers.Dense(5, activation='softmax')(net)
    
#     model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
#     model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
#     return model
    
# text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
# preprocessor = hub.KerasLayer("http://tfhub.dev/tensorflow/albert_en_preprocess/3")
# encoder_inputs = preprocessor(text_input)

class Model:
    preprocessor = hub.KerasLayer("http://tfhub.dev/tensorflow/albert_en_preprocess/3")

    def __init__(self):
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        encoder_inputs = self.preprocessor(text_input)
        encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_base/2", trainable=True)
        outputs = encoder(encoder_inputs)
        net = tf.keras.layers.Dropout(0.1)(outputs['pooled_output'])
        net = tf.keras.layers.Dense(1, dtype=tf.float32)(net)
        self.model = tf.keras.Model(text_input, net)

    def train(self, train_data, train_labels, epochs=5):
        """trains the model using the data given"""
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(train_data, train_labels, epochs=epochs)

    def evaluate(self, test_data, test_labels):
        """evaluates the model given the test data and labels"""
        self.model.evaluate(test_data, test_labels)

if __name__ == '__main__':

    #creating instance of the model class to be used later and a train_portion for later use 
    model = Model()
    train_portion = 0.9 #how much of the data will go into training(0-1)

    #fetching the data from the csv files
    notes_data = pd.read_csv('notes_data.csv')['notes commands'].to_numpy(dtype=np.str)
    calories_data = pd.read_csv('calories_data.csv', delimiter=',')['calories commands'].to_numpy(dtype=np.str)

    #labeling and combining the data together
    data = np.array(notes_data.tolist() + calories_data.tolist(), dtype=np.str)
    labels = np.array(np.ones(notes_data.shape).tolist() + np.zeros(calories_data.shape).tolist())

    #mixing/shuffling the data the data
    shuffled_data, shuffled_labels = shuffle(data, labels)

    #split the data into train and test data
    train_data = shuffled_data[:len(shuffled_data)]
    train_labels = shuffled_labels[:len(shuffled_labels)]
    test_data = shuffled_data[len(shuffled_data)+1:]
    test_labels = shuffled_labels[len(shuffled_labels)+1:]

    #training the model(only the output layer currently will be trained)
    model.train(train_data, train_labels)

    #evaulating the model
    model.evaluate(test_data, test_labels)