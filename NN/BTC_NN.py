AVG_OFFSETS = list(range(2, 18))
OFFSET = 2
RESULT_OFFSET = 5
CHECKPOINT_PATH = 'training3/'
SAVE_PATH = 'training3/'

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.utils import normalize
import tensorflow as tf
import matplotlib.pyplot as plt

def create_trainset(file, avg_offsets = [10, 20, 35], offset = 15, result_offset = 7):
    global train_array
    with open(file, 'r') as train_file:
        for line in train_file:
            train_array = line.split(',')
    
    train_array = list(map(float, train_array))

    result_set = np.zeros(shape = (len(train_array) - max( max(avg_offsets), offset ) - result_offset , 1))
    print(np.shape(result_set))
    for num in range(len(result_set)):
        summ = sum(train_array[num + max(max(avg_offsets), offset) : num + max(max(avg_offsets), offset) + result_offset]) / RESULT_OFFSET
        if train_array[num + max(max(avg_offsets), offset)] - summ > 0:
            result_set[num, 0] = 1
        else:
            result_set[num, 0] = 0
    
    train_set = np.zeros(( len(train_array) - max(max(avg_offsets), offset) - result_offset, len(avg_offsets) + offset ))
    for position in range(len(train_set)):
        train_set[position] = fill_single_pos(train_array, avg_offsets, offset, position + max(max(avg_offsets), offset))
    
    #result_set /= np.amax(train_set)
    #train_set /= np.amax(train_set)
    np.savetxt('trainset.txt', train_set)
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

x_train, y_train = create_trainset('btc_hour.txt', AVG_OFFSETS, OFFSET, RESULT_OFFSET)
cp_callback = tf.keras.callbacks.ModelCheckpoint(SAVE_PATH, save_weights_only = True, verbose = 1)
nn = create_network()

nn.load_weights(CHECKPOINT_PATH)
nn.fit(x_train[:24000], y_train[:24000], epochs = 10000, callbacks = [cp_callback])

starting_point = 24020
end_point = 25122
output = np.zeros(end_point - starting_point)
mistakes2 = np.zeros(end_point - starting_point)
balance = 100
btc = 0
balance_array = []
btc_array = []
prev = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for num in range(starting_point, end_point):
    prediction = nn.predict(np.array([fill_single_pos(train_array, AVG_OFFSETS, OFFSET, num)]))
    if prediction[0, 0] not in prev:
        if prediction[0, 0] < 0.00005:
            balance /= 2
            btc += balance / train_array[num] * 0.9975
        elif prediction[0, 0] > 0.99995:
            btc /= 2
            balance += btc * train_array[num] * 0.9975
    prev[num % len(prev)] = prediction[0, 0]
    balance_array.append(balance)
    btc_array.append(btc)
    output[num - starting_point] = prediction[0, 0]
    mistakes2[num - starting_point] = train_array[num] - train_array[num + RESULT_OFFSET]

print(balance + btc * train_array[end_point])

plt.plot([x for x in range(len(mistakes2))], mistakes2, color = 'blue')
plt.savefig('mistakes.png')
plt.close()
plt.plot([x for x in range(len(output))], output, color = 'red')
plt.savefig('predictions.png')
plt.close()
plt.plot([x for x in range(len(btc_array))], btc_array, color = 'orange')
plt.savefig('btc.png')
plt.close()
plt.plot([x for x in range(len(balance_array))], balance_array, color = 'green')
plt.savefig('balance.png')