import sys

import numpy as np

from foucault import get_dataset, project_swing_on_floor, normalize_vectors

if __name__ == '__main__':
    args = sys.argv[1:]

    series = args[0]
    name = args[1]

    df = get_dataset(series, name)

    t = df['t']

    x = df['x']
    y = df['y']
    z = df['z']

    vectors, timestamps = project_swing_on_floor(x, y, t, [0, 200])

    normalized_vectors = normalize_vectors(vectors)

    angle_radians = np.arccos(np.dot(normalized_vectors[0], normalized_vectors[1]))

    angle_degrees = np.degrees(angle_radians)

    time_difference = timestamps[1] - timestamps[0]

    print("Angle difference: ", angle_degrees)
    print("Time difference: ", time_difference)
