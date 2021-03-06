import matplotlib.pyplot as plt
import numpy as np


def print_progress_bar(iteration, total, prefix='', suffix='', length=100, timer=0.0, fill='='):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        length      - Optional  : character length of bar (Int)
        timer       - Optional  : Process timer (int)
        fill        - Optional  : bar fill character (Str)
    """
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '>' * int((length - filled_length) > 0) + '.' * (length - filled_length - 1)
    print('\r%s [%s] %1.1fs %s' % (prefix, bar, timer, suffix), end='\r')


def r_squared(y_true, y_pred):
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    ss_res = np.sum(np.square(y_true - np.abs(y_pred)))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - ss_res / ss_tot


def gen_image(arr, shape=(28, 28)):
    two_d = (np.reshape(arr, shape) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    return plt
