from collections import deque
import numpy as np


class SimpleSampleDatabase:
    def __init__(self, size):
        self.size = size
        self.data_x = deque(maxlen=size)
        self.data_y = deque(maxlen=size)

    def add_data(self, data_x, data_y):
        self.data_x.extend(data_x)
        self.data_y.extend(data_y)

    def get_data(self):
        data_x = np.vstack(self.data_x)
        data_y = np.vstack(self.data_y)
        return data_x, data_y
