# SHOWING THE FIRST 30 SECONDS OF DATA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
import mne
import os

# get directory path
dir_path = os.path.dirname(os.path.realpath(__file__))
psg_data = mne.io.read_raw_edf(dir_path+'/../../data/raw/SN001.edf', preload=True)  # load data into memory

print(psg_data.ch_names)
print(psg_data.annotations)
print(psg_data.times)
print(psg_data.times)
print(psg_data.info)


# get the first n seconds of data
n = 30
start, stop = psg_data.time_as_index([0, n])  # get indices for the first 30 seconds

# get the data and times for this interval
data, times = psg_data[:, start:stop]  # get the data and times for this interval

# Plot the data
plt.figure(figsize=(10, 5))
for channel in range(data.shape[0]):
    # if psg_data.ch_names[channel] == 'ECG':
    plt.plot(times, data[channel, :], label=psg_data.ch_names[channel])

plt.legend(loc='upper right')
plt.xlabel('Time (s)')
plt.ylabel('Signal')
plt.title('First 30 seconds of data')
plt.show()



def plotChannel(data, channel_name):
    channel_index = data.ch_names.index(channel_name)
    start, stop = data.time_as_index([0, 30])  # plot first 30 seconds
    channel_data, times = data[channel_index, start:stop]
    
    plt.figure(figsize=(10, 5))
    plt.plot(times, channel_data.T, label=channel_name)
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.title(f'Data from {channel_name} channel')
    plt.show()

# usage
channels = psg_data.ch_names
plotChannel(psg_data, channels[1])