def load_position(file, position, amount = 1):
    for line in file:
        return [int(eval(i)) for i in line.split(',')[position + 1 - amount : position + 1]]