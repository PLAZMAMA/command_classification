import tensorflow as tf
import tensorflow_hub as hub
from Model import Model
import sys

#when running this file use input the name of the model that should be used as a command line argument
#ex: python model_predict.py <model_name>

if __name__ == '__main__':
    #loading the model from the given model name in the command line argument that was given when running this file
    try:
        model = Model(tf.keras.models.load_model(f'models/{sys.argv[1]}', custom_objects={'KerasLayer': hub.KerasLayer}))

    except Exception as e:
        raise Exception('command line argument was not given(the model name was not given, please put the name of the file that the script should use)')

    #while loop that checks for when the prediction_input file gets updated,
    #then, it takes it text, cleans the file and prints the prediction to the screen
    while(True):
        with open('prediction_input.txt', 'r') as f:
            sentence = f.read()
            if len(sentence) > 0:
                print(model.predict(sentence))