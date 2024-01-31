import csv
import numpy as np


def from_csv(file_name):
    gates = []
    with open(file_name) as file:
        reader = csv.reader(file)
        for row in reader:
            gates.append(np.array([float(x) for x in row]))
    return np.array(gates)
