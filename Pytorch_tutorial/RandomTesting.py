import numpy as np


tensor = [
    [5, 4, 7, 2, 4],
    [4, 3, 4, 2, 1],
]

if __name__ == '__main__':
    indices = np.array([3, 1])
    tensor = np.array(tensor)
    print(tensor[np.where(indices)+(indices,)])