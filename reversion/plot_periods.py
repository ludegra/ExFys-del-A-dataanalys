import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import sys
import os

from reversion import get_all_periods

if __name__ == '__main__':
    args = sys.argv[1:]
    
    series = args[0]

    periods_a, periods_b = get_all_periods(series, 'y')


    periods_a.sort(key=lambda x: x['distance'])
    periods_b.sort(key=lambda x: x['distance'])


    ta = [a['period'] for a in periods_a]
    da = [a['distance'] for a in periods_a]

    tb = [a['period'] for a in periods_b]
    db = [a['distance'] for a in periods_b]

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

    ax[0].plot(da, ta, 'o-', label='TA')
    ax[0].plot(db, tb, 'o-', label='TB')

    ax[1].plot(da, np.array(ta) - np.array(tb), 'o-', label='TA - TB')

    ax[1].hlines([0], min(da), max(da), linestyle='--', color='black')

    ax[0].legend()

    ax[0].set_xlabel('Distance')
    ax[0].set_ylabel('Period time')

    ax[1].set_xlabel('Distance')
    ax[1].set_ylabel('Difference in period time')

    fig.tight_layout()

    plt.show()