import networks.direction_predictor
from global_variables import *
from matplotlib import pyplot as plt
import utils.trainset_utils as tu
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
import numpy as np

class Networks:
    dp = networks.direction_predictor

class Network:
    name: str
    trainset_x: np.array
    trainset_y: np.array
    callback: ModelCheckpoint
    network: Sequential

    def __init__(self, network, name = ''):
        self.net_type = network
        self.network = network.create_network()
        self.name = name
        self.callback = None
        self.offset_min = 0
        self.offset_max = 1
    
    def load_trainset(self, filename):
        with open(f'{TRAINSETS_FOLDER}{filename}.txt') as f:
            self.trainset_x, self.trainset_y = self.net_type.create_trainset(f)

            if self.offset_max > np.size(self.trainset_x, 0) or self.offset_max == 1:
                self.offset_max = np.size(self.trainset_x, 0)
    
    def load_weights(self, filename):
        self.network.load_weights(f'{CHECKPOINTS_FOLDER}{filename}')
        
    def set_callback(self, name, verbose):
        self.callback = ModelCheckpoint(f'{CHECKPOINTS_FOLDER}{name}', save_weights_only = True, verbose = verbose)
    
    def set_trainset_offsets(self, min = 0, max = 1):
        self.offset_min = min
        self.offset_max = max
        print(np.size(self.trainset_x[self.offset_min : self.offset_max], 0))
    
    def train(self, epochs = 1000):
        self.network.fit(self.trainset_x[self.offset_min : self.offset_max], self.trainset_y[self.offset_min : self.offset_max], epochs = epochs, callbacks = [self.callback] if self.callback else [])
    
    def predict(self, position):
        data = self.trainset_x[position]
        prediction = self.network.predict(np.array([data]))
        pos_from_file = np.array(tu.load_position(open(f'{TRAINSETS_FOLDER}btc_hour.txt', 'r'), position, amount = 20))
        pos_from_file = np.append( pos_from_file, np.array( [( pos_from_file[np.size( pos_from_file, 0 ) - 1 ] + 50 ) if prediction[0, 0] > 0.5 else ( pos_from_file[ np.size(pos_from_file, 0) - 1 ] - 50 ) for _ in range(5) ]))
        plt.plot(pos_from_file, color = 'blue')
        plt.show()

class Control_panel:

    def __init__(self):
        self.networks = {}
    
    def __getitem__(self, key):
        return self.networks[key]

    def new_network(self, type, name):
        self.networks[name] = (Network(type, name))

cp1 = Control_panel()
cp1.new_network(Networks.dp, 'DP')
cp1['DP'].load_weights('training3/')
cp1['DP'].load_trainset('btc_hour')
cp1['DP'].set_trainset_offsets(max = 24000)
cp1['DP'].train(1)
cp1['DP'].predict(25100)
