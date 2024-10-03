import sys
import itertools

import matplotlib.pyplot as plt
import numpy as np

from foucault import get_dataset, project_swing_on_floor, normalize_vectors, sanitize_vectors, angle_between_vectors

def sliding_window(iterable, n):
    iters = itertools.tee(iterable, n)
    for i, it in enumerate(iters):
        next(itertools.islice(it, i, i), None)
    return zip(*iters)

if __name__ == '__main__':
    args = sys.argv[1:]

    series = args[0]
    name = args[1]

    df = get_dataset(series, name)

    t = df['t']

    x = df['x']
    y = df['y']
    z = df['z']

    vectors, timestamps = project_swing_on_floor(x, y, t, itertools.count(start=1, step=1))

    normalized_vectors = normalize_vectors(vectors)

    normalized_vectors, timestamps = sanitize_vectors(normalized_vectors, timestamps)

    frequencies = []  

    window_size = 1

    for v, t in zip(sliding_window(normalized_vectors, 2), sliding_window(timestamps, 2)):
        v1 = v[0]
        v2 = v[1]

        t1 = t[0]
        t2 = t[1]

        delta_angle = angle_between_vectors(v1, v2)
        delta_time = t2 - t1

        frequency = delta_angle / delta_time

        frequencies.append(frequency)

    frequencies = np.array(frequencies)

    time_threshold = 1500

    cutoff_index = np.searchsorted(timestamps, time_threshold)

    early_frequencies = frequencies[:cutoff_index]
    early_timestamps = timestamps[:cutoff_index]

    mean_frequency = np.mean(early_frequencies)
    median_frequency = np.median(early_frequencies)

    print("Mean frequency: ", mean_frequency)
    print("Median frequency: ", median_frequency)
    
    plt.plot(early_timestamps, early_frequencies, "o")

    plt.hlines(mean_frequency, min(early_timestamps), max(early_timestamps), color="red", linestyle="dashed", label="Mean")
    plt.hlines(median_frequency, min(early_timestamps), max(early_timestamps), color="red", linestyle="dotted", label="Median")

    plt.legend()

    plt.show()
    

