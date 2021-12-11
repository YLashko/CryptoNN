import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.utils import normalize
import tensorflow as tf
import matplotlib.pyplot as plt

AVG_OFFSETS = list(range(2, 18))
OFFSET = 2
RESULT_OFFSET = 5

