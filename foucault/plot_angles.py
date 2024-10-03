import sys
import itertools

import numpy as np
import matplotlib.pyplot as plt

from foucault import get_dataset, get_plane_normal_vectors, angle_between_vectors, REFERENCE_VECTOR

if __name__ == '__main__':
    args = sys.argv[1:]

    series = args[0]
    name = args[1]

    df = get_dataset(series, name)

    n, t = get_plane_normal_vectors(df, 'z', itertools.count(start=0, step=2))

    angles = angle_between_vectors(n, REFERENCE_VECTOR)

    degrees = np.degrees(angles)

    plt.figure(figsize=(15, 5))

    plt.plot(t, degrees, "o", label="Angle against the xz-plane")

    plt.show()
