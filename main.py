import tensorflow as tf
import mne
import math

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
    model.add(tf.keras.layers.Dense())
    pass

def main():
    raw_data, scoring = read_data()

    sample_duration = 30 # 30 seconds
    # n_samples = math.floor(raw_data.times[-1]/sample_duration) # number of 30-second windows, rounded down
    interval_start = 0
    score_index = 0
    while True: #divide data into 30-second chunks. times[-1] is the last time-moment recorded
        interval_stop = interval_start + sample_duration
        if interval_stop > raw_data.times[-1]: break

        start, stop = raw_data.time_as_index([interval_start, interval_stop]) # get indexes from time
        data, times = raw_data[:, start:stop] # every sample from evey channel in start->stop time interval

        interval_start += sample_duration
        print(scoring.description[score_index])
        score_index += 1


if __name__ == "__main__":
    main()