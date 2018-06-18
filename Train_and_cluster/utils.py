import numpy as np
import glob
import os


def broadcasting_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    return a[S * np.arange(nrows)[:, None] + np.arange(L)]


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n), writeable=False)


def class_imbalance(arr):
    arr = arr.ravel()
    assert(len(arr) == np.count_nonzero(arr == 0) + np.count_nonzero(arr == 1))
    return np.count_nonzero(arr) / len(arr)


def get_unstrided_data():
    prefix = 'data'
    train_features_directory = os.path.join(prefix, 'speech', '*.npy')
    train_labels_directory = os.path.join(prefix, 'peaks', '*.npy')

    feature_files = sorted(glob.glob(train_features_directory))
    label_files = sorted(glob.glob(train_labels_directory))

    x_data = np.concatenate([np.load(f) for f in feature_files])
    y_data = np.concatenate([np.load(f) for f in label_files])

    assert(len(x_data) == len(y_data))
    N = len(x_data)
    split_point = int(0.8 * N)

    x_train = x_data[:split_point]
    y_train = y_data[:split_point]
    x_test = x_data[split_point:]
    y_test = y_data[split_point:]

    return (x_train, y_train), (x_test, y_test)
