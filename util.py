import pickle


def store_data(data, root):
    with open(root, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_data(root):
    with open(root, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


