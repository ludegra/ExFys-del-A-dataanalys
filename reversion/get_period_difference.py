import sys

import numpy as np

from reversion import get_dataset, get_mean_period


if __name__ == '__main__':
    args = sys.argv[1:]

    series = args[0]
    distance = args[1]

    dfa = get_dataset(series, 'A', distance)
    dfb = get_dataset(series, 'B', distance)

    period_a = get_mean_period(dfa, 'y')
    period_b = get_mean_period(dfb, 'y')

    print("Period difference: " + str(np.abs(period_a - period_b)))


