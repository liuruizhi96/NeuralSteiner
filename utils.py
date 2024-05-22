import numpy as np
import os
import pickle

def save_as_pickle(filename, data):
    completeName = os.path.join("./ispd18_data/", filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)
    return 0


def load_pickle(filename):
    completeName = os.path.join("./ispd18_data/", filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data


# 比较两个数的大数
def max_num(a,b):
    if a >= b:
        return a,b
    else:
        return b,a

# 比较两个数的小数
def min_num(a,b):
    if a <= b:
        return a
    else:
        return b