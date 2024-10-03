import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import sys
import os

def get_dataset(series, pivot, distance):
    file_name = f"Reversion_T{pivot}_{distance}.tsv"
    file_path = f"./data/serie_{series}/{file_name}"

    df = pd.read_csv(file_path, sep='\t', names=['index', 't', 'x', 'y', 'z'])

    return df

def get_mean_period(df, axis):
    peaks, _ = sp.signal.find_peaks(df[axis])
    peak_times = df['t'][peaks]

    periods = np.diff(peak_times)
    return np.mean(periods)

def get_all_periods(series, axis):
    series_dir = f"./data/serie_{series}"

    periods_a = []
    periods_b = []

    for file in os.listdir(series_dir):
        split = file.split('_')

        pivot = split[1][-1]
        distance = split[2].split('.')[0]

        df = get_dataset(series, pivot, distance)

        period = get_mean_period(df, 'y')

        if pivot == 'A':
            periods_a.append({'period': period, 'distance': float(distance.replace(',', '.'))})

        elif pivot == 'B':
            periods_b.append({'period': period, 'distance': float(distance.replace(',', '.'))})

    return periods_a, periods_b





# plt.plot(t, y, label='y')
# plt.vlines(peak_times, min(y), max(y), linestyles='dashed', label='peaks') 

# plt.legend()

# plt.show()