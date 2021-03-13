import tensorflow as tf
import sys

#when running this file use input the name of the model that should be used as a command line argument
#ex: python model_predict.py <model_name>

if __name__ == '__main__':
    #loading the model from the given model name in the command line argument that was given when running this file
    try:
        model = tf.keras.load_model(f'models/{sys.argv[1]}')
    except Exception as e:
        raise Exception('command line argument was not given')