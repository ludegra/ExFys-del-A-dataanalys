import sys
import itertools

import numpy as np
import matplotlib.pyplot as plt

from foucault import get_dataset, project_swing_on_floor, normalize_vectors, sanitize_vectors

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

    normalized_vectors, timestams = sanitize_vectors(normalized_vectors, timestamps)

    U = normalized_vectors[:, 0]
    V = normalized_vectors[:, 1]

    X = np.zeros(len(normalized_vectors))
    Y = np.zeros(len(normalized_vectors))

    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)

    plt.plot(X, Y)

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    plt.show()
