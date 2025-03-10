import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
import os

class Model:
    preprocessor = hub.KerasLayer("http://tfhub.dev/tensorflow/albert_en_preprocess/3")
    classes = ['notes command', 'calories command', 'random']
    trained = False

    def __init__(self, model=None, model_size='base'):
        #checking if a model was given and loading the given model or creating a model with random weights(which usually is used for training)
        if model:
            self.model = model
        
        else:
            #model arcitecture(creating an instance of a keras model using the defined layers)
            text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
            encoder_inputs = self.preprocessor(text_input)
            encoder = hub.KerasLayer(f'https://tfhub.dev/tensorflow/albert_en_{model_size}/3')
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
        biggest_model_num = 0
        for file_name in file_names:
            ending_point = len(file_name)-1

            #gets the ending point of the model number
            while(file_name[ending_point] != '_'):
                ending_point -= 1

            if int(file_name[5:ending_point]) > biggest_model_num:
                biggest_model_num = int(file_name[5:ending_point])
        
        return biggest_model_num + 1


    def train(self, train_data, train_labels, epochs=5, batch_size=16):
        """trains the model using the data given"""
        #getting the next number that the model should be saved as
        next_model_num = self.get_next_file_name()
                
        #creating a model check point which creates checkpoints and saves them at each epoch, and also training it
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(os.pardir, f'/code/models/bert_{next_model_num}' + '_{epoch:02d}-{loss:.2f}.hdf5'), save_best_only=True, monitor='loss')
        self.model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, callbacks=model_checkpoint_callback)
        trained = True

    def evaluate(self, test_data, test_labels, batch_size=16):
        """evaluates the model given the test data and labels"""
        results = self.model.evaluate(test_data, test_labels, batch_size=batch_size)
        print(f'Total Loss: {results[0]}, Total Accuracy: {results[1] * 100}%')
    
    def predict(self, prediction_data, prediction_output='class'):
        """
        uses the current model to predict/classify which class the prediction data falls into
        prediction_output: 'class'|'array'
            prediction_output can be either class which means that the output will be a string name of the class,
            or array which means that the output will be an array with the size of the number of classes and contains floats
        """
        #getting the prediction of the given data by calling the current model
        result = self.model.predict([prediction_data])

        #handling the different options of the format the output should be returned as
        if prediction_output == 'class':
            return self.classes[np.argmax(result)]
        
        elif prediction_output == 'array':
            return result
        
        else:
            raise RuntimeError('"prediction_output" argument wasnt given one of the two options listed in the function description')
