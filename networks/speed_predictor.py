from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.utils import normalize
import tensorflow as tf

def create_network():
    net = Sequential()
    net.add(Flatten())
    net.add(Dense(120, activation = tf.nn.sigmoid))
    net.add(Dense(120, activation = tf.nn.sigmoid))
    net.add(Dense(1, activation = tf.nn.sigmoid))

    net.compile(optimizer = 'adam',
    loss = 'mean_absolute_error')

    return net