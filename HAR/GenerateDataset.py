import os
import numpy as np
import pandas as pd

def generate_dataset(filepath: str = os.path.join('Datasets', 'UCI HAR Dataset'), folder: str = 'Train') -> pd.DataFrame:
    f = open(os.path.join(filepath, 'features.txt'), 'r')
    lines = f.readlines()
    f.close()
    features = []
    exists = {}

    for line in lines:
        new_line = line.replace('\n', '').replace('-', '_')[line.find(' ') + 1:]
        if new_line in exists:
            # for features like fBodyAcc_bandsEnergy()_1,8 with multiple occurrences
            features.append(new_line + "_" + str(exists[new_line]))
            exists[new_line] += 1
        else:
            features.append(new_line)
            exists[new_line] = 1

    dataset_path = os.path.join(filepath, folder)
    X_file = open(os.path.join(dataset_path, f'X_{folder}.txt'))
    f.close()
    lines = X_file.readlines()
    X = []
    for line in lines:
        row = np.array(line.split(), dtype=np.float64)
        X.append(row)
    X = np.array(X)
    X = pd.DataFrame(X, columns=features)
    
    y_file = open(os.path.join(dataset_path, f'y_{folder}.txt'))
    f.close()
    lines = y_file.readlines()
    y = []
    for line in lines:
        y.append(line)
    y = np.array(y, dtype=np.int32) - 1
    X['y'] = pd.Series(y)
    return X