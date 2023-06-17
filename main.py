import tensorflow as tf
import mne
import math
import numpy as np
from progressbar import ProgressBar, Percentage, Bar, FormatLabel

def read_data():
    raw_data = mne.io.read_raw_edf('./datasets/SN001.edf', preload=True)
    scoring = mne.read_annotations('./datasets/SN001_sleepscoring.edf')
    return raw_data, scoring

"""identified problems:
    which layers to add to the model?
    how many nodes in a layer?"""
def define_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten()) # turns multidimensional data into a vector
    model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(5, activation=tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def map_classes(raw_labels: list[str]) -> list[int]:
    numeric_labels = [] # W, N1, N2, N3, R
    pbar = ProgressBar(widgets=[Percentage(), Bar(), FormatLabel("Mapping classes...")], maxval=len(raw_labels)).start()
    n_done = 0
    for label in raw_labels:

        match label:
            case 'Sleep stage W':
                numeric_labels.append(0)
            case 'Sleep stage N1':
                numeric_labels.append(1)
            case 'Sleep stage N2': 
                numeric_labels.append(2)
            case 'Sleep stage N3':
                numeric_labels.append(3)
            case 'Sleep stage R':
                numeric_labels.append(4)
        n_done +=1
        pbar.update(n_done)
    pbar.finish()
    return numeric_labels



def prepare_training_dataset(raw_data, raw_labels):
    X = []
    Y = map_classes(raw_labels.description) # each element here is a label, 0-4
    
    sample_duration = 30 # 30 seconds
    n_intervals = math.floor(raw_data.times[-1]/sample_duration) # number of 30-second windows, rounded down

    pbar = ProgressBar(widgets=[Percentage(), Bar(), FormatLabel("Generating training data...")], maxval=n_intervals).start()
    
    interval_start = 0
    score_index = 0
    downsampling = 10 # every 10th element
    while True: #divide data into 30-second chunks. times[-1] is the last time-moment recorded
        interval_stop = interval_start + sample_duration
        if interval_stop > raw_data.times[-1]: break

        start, stop = raw_data.time_as_index([interval_start, interval_stop]) # get indexes from time
        data, times = raw_data[:, start:stop] # every sample from evey channel in start->stop time interval
        downsampled_data = data[:, ::downsampling] # feed this into the neural network, 
        X.append(downsampled_data)
        #[:, ::downsampling] - get every channel, but every nth sample from each channel
        # every interval is a 8x768 "image" (with downsampling of 10), which yields 6144 points per sample

        interval_start += sample_duration
        score_index += 1
        pbar.update(score_index) # score index is coincidentally hte same value as the progress bar needs

    pbar.finish()

    return X, Y


def main():
    raw_data, raw_scoring = read_data()
    X, Y = prepare_training_dataset(raw_data, raw_scoring)
    model = define_model()
    model.fit(X, Y, epochs=2)
    # TODO Keras jest wrażliwy na typy tablic - X to jest lista tablic dwuwymiarowych.
    # Y to lista - tablica jednowymiarowa. Trzeba zrobić tak, żeby każda tablica/lista były w typach "numpy-owych"
    # czyli X to by była np.array zawierające tablice wielowymiarowe np.ndarray.
    # tak samo, Y powinno być zwykłym mp.array, jednowymiarowa tablica.



if __name__ == "__main__":
    main()


"""TODO:
    find a way to load channel data from EDF to an array. Each interval is a 2D array - 8 rows, one for each channel, 30*256 samples per channel.
load this data properly into np arrays.


    Normalise X data
    
    consider treating our data as an image - its around 8x781 per interval"""