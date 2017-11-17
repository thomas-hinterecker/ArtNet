import numpy as np

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', length = 100, timer = 0.0, fill = '='):
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
    filledLength = int(length * iteration // total)
    bar = fill * (filledLength) + '>' * int((length - filledLength) > 0) + '.' * (length - filledLength -1)
    print('\r%s [%s] %1.1fs %s' % (prefix, bar, timer, suffix), end = '\r')
    #if iteration == total: 
    #    print()

def r_squared(y_true, y_pred):
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    ss_res = np.sum(np.square(y_true - np.abs(y_pred)))
    ss_tot= np.sum(np.square(y_true - np.mean(y_true)))

    return 1 - ss_res / ss_tot