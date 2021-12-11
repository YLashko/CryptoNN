from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.utils import normalize
import tensorflow as tf
import numpy as np

def create_trainset(file, avg_offsets = list(range(3, 18)), offset = 3, result_offset = 3):
    global train_array
    for line in file:
        train_array = line.split(',')
    
    train_array = list(map(float, train_array))

    result_set = np.zeros(shape = (len(train_array) - max( max(avg_offsets), offset ) - result_offset , 1))
    print(np.shape(result_set))
    for num in range(len(result_set)):
        summ = sum(train_array[num + max(max(avg_offsets), offset) : num + max(max(avg_offsets), offset) + result_offset]) / result_offset
        if train_array[num + max(max(avg_offsets), offset)] - summ > 0:
            result_set[num, 0] = 1
        else:
            result_set[num, 0] = 0
    
    train_set = np.zeros(( len(train_array) - max(max(avg_offsets), offset) - result_offset, len(avg_offsets) + offset ))
    for position in range(len(train_set)):
        train_set[position] = fill_single_pos(train_array, avg_offsets, offset, position + max(max(avg_offsets), offset))
    
    #result_set /= np.amax(train_set)
    #train_set /= np.amax(train_set)
    print(train_set[20000])
    print(result_set[20000])
    return train_set, result_set

def fill_single_pos(train_array, avg_offsets, offset, position):

    pos = np.zeros((len(avg_offsets) + offset))
    
    for num in range(len(avg_offsets)):
        pos[num] = (train_array[position] - sum(train_array[position - avg_offsets[num] + 1 : position + 1]) / avg_offsets[num])
    
    for num in range(offset):
        pos[num + len(avg_offsets)] = (train_array[position] - train_array[position - num])

    return pos

def create_network():
    net = Sequential()
    net.add(Flatten())
    net.add(Dense(120, activation = tf.nn.sigmoid))
    net.add(Dense(120, activation = tf.nn.sigmoid))
    net.add(Dense(1, activation = tf.nn.sigmoid))

    net.compile(optimizer = 'adam',
    loss = 'mean_absolute_error')

    return net