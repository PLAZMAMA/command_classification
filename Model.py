import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
import os

class Model:
    preprocessor = hub.KerasLayer("http://tfhub.dev/tensorflow/albert_en_preprocess/3")
    classes = ['notes command', 'calories command', 'random']
    trained = False

    def __init__(self):
        #model arcitecture(creating an instance of a keras model using the defined layers)
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        encoder_inputs = self.preprocessor(text_input)
        encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_base/2")
        outputs = encoder(encoder_inputs)
        net = tf.keras.layers.Dense(256)(outputs['pooled_output'])
        net = tf.keras.layers.Dropout(0.3)(net)
        net = tf.keras.layers.Dense(3, activation='softmax', dtype=tf.float32)(net)
        self.model = tf.keras.Model(text_input, net)
    
    def get_next_file_name(self):
        """gets the next number that should be used for the model name"""
        #getting all the file names from the models directory
        file_names = os.listdir(os.path.join(os.pardir, '/code/models'))

        #loop that goes throught each file name and its characters and get the largest file number
        next_model_num = 0
        for file_name in file_names:
            ending_point = len(file_name)-1
            while(file_name[ending_point] != '_'):
                ending_point -= 1

            if int(file_name[5:ending_point]) > next_model_num:
                next_model_num = int(file_name[5:ending_point])
        
        return next_model_num + 1


    def train(self, train_data, train_labels, epochs=5):
        """trains the model using the data given"""
        #getting the next number that the model should be saved as
        next_model_num = self.get_next_file_name()
                
        #giving the preprocessor the data, compiling the model and training it
        encoder_inputs = self.preprocessor(train_data)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        tf.keras.callbacks.ModelCheckpoint(os.path.join(os.pardir, f'/code/models/bert_{next_model_num}_{epoch:02d}.hdf5'), save_best_only=True)
        self.model.fit(train_data, train_labels, batch_size=16, epochs=epochs)
        trained = True

    def evaluate(self, test_data, test_labels):
        """evaluates the model given the test data and labels"""
        results = self.model.evaluate(test_data, test_labels, batch_size=16)
        print(f'Total Loss: {results[0]}, Total Accuracy: {results[1] * 100}%')
    
    def predict(self, prediction_data, model_num=None, epoch=5):
        """uses the model to predict/classify which class the prediction data falls into"""
        if model_num:
            try:
                model = tf.keras.models.load_model(os.path.join(os.pardir, f'/code/models/bert_{model_num}_{epoch}.hdf5'))

            except Exception as e:
                raise Exception()

            result = model.predict(prediction_data)

        elif trained:
            result = self.model.predict(prediction_data)

        else:
            raise RuntimeError('"model num" argument was not passed and this instance of the model class hasnt run the train function')

        return self.classes[np.argmax(result)]