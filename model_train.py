from Model import Model
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

if __name__ == '__main__':
    #creating instance of the model class to be used later and a train_portion for later use 
    model = Model()
    train_portion = 0.9 #how much of the data will go into training(0-1)

    #fetching the data from the csv files
    notes_data = pd.read_csv('notes_data.csv')['notes commands'].to_numpy(dtype=np.str)
    calories_data = pd.read_csv('calories_data.csv', delimiter=',')['calories commands'].to_numpy(dtype=np.str)
    smp_data = pd.read_csv('smp.csv')['commands'].to_numpy(dtype=np.str)

    #labeling and combining the data together
    data = np.array(notes_data.tolist() + calories_data.tolist() + smp_data.tolist(), dtype=np.str)
    labels = np.array([[1, 0, 0]] * notes_data.shape[0] + [[0, 1, 0]] * calories_data.shape[0] + [[0, 0, 1]] * smp_data.shape[0])

    #mixing/shuffling the data the data
    shuffled_data, shuffled_labels = shuffle(data, labels)

    #split the data into train and test data
    train_test_index = int(len(shuffled_data) * train_portion) #the index where the test data start and the training data stops
    train_data = shuffled_data[:train_test_index]
    train_labels = shuffled_labels[:train_test_index]
    test_data = shuffled_data[train_test_index + 1:]
    test_labels = shuffled_labels[train_test_index + 1:]

    #training the model(only the output layer currently will be trained)
    model.train(train_data, train_labels)

    #evaulating the model
    model.evaluate(test_data, test_labels)