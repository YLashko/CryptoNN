from global_variables import TRAINSETS_FOLDER, RAW_FOLDER
INPUT_FILE_NAME = f'{RAW_FOLDER}Bitstamp_BTCUSD_1h.csv'
OUTPUT_FILE_NAME = f'{TRAINSETS_FOLDER}btc_hour1.txt'
COLUMN = 3
PART = [0, 28000]
REVERSE = True

with open(INPUT_FILE_NAME, 'r') as file:
    array = [line.replace('\n', '').split(',')[COLUMN] for line in file]

with open(OUTPUT_FILE_NAME, 'w') as file:
    if REVERSE:
        for i in range(PART[1] - 1, PART[0] - 1, -1):
            file.write(f'{array[i]},')
    else:
        for i in range(PART[0], PART[1]):
            file.write(f'{array[i]},')