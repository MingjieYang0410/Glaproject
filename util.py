import pickle
import random

def store_data(data, root):
    with open(root, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_data(root):
    with open(root, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

def return_rand():
    i = random.randint(1, 10)
    return i


def calc_auc(raw_arr):
    """Summary
    Args:
        raw_arr (TYPE): Description
    Returns:
        TYPE: Description
    """

    arr = sorted(raw_arr, key=lambda d:d[0], reverse=True)
    pos, neg = 0, 0
    for record in arr:
        if record[1] == 1:
            pos += 1
        else:
            neg += 1

    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if record[1] == 1:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp/neg, tp/pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        if x != prev_x:
            auc += ((x - prev_x) * (y + prev_y) / 2.)
            prev_x = x
            prev_y = y

    return auc