from utils.trainset_utils import load_position

print(load_position(open('trainsets/btc_hour1.txt', 'r'), 20000, 5))