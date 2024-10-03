import pandas as pd
import scipy as sp
import numpy as np

REFERENCE_VECTOR = [0, 1, 0]

def sign(x):
    -1 if x < 0 else 1

def get_dataset(series, name):
    file_name = f"Foucault_{name}.tsv"
    file_path = f"./data/serie_{series}/{file_name}"

    df = pd.read_csv(file_path, sep='\t')

    df.columns = ['index', 't'] + [f'col_{i}' for i in range(2, len(df.columns))]

    x_columns = df.iloc[:, 2::3]
    y_columns = df.iloc[:, 3::3]
    z_columns = df.iloc[:, 4::3]

    df['x'] = x_columns.sum(axis=1)
    df['y'] = y_columns.sum(axis=1)
    df['z'] = z_columns.sum(axis=1)

    df = df[~((df['x'] == 0) & 
                          (df['y'] == 0) & 
                          (df['z'] == 0))]

    return df[['t', 'x', 'y', 'z']]

def get_plane_normal_vectors(df, height_axis, peak_indices):
    normal_vectors = []
    timestamps = []

    h = df[height_axis]

    floor_axes = ['x', 'y', 'z']
    floor_axes.remove(height_axis)

    s1 = df[floor_axes[0]]
    s2 = df[floor_axes[1]]

    peaks, _ = sp.signal.find_peaks(h)
    peak_times = df['t'][peaks].values

    for peak_index in peak_indices:
        if np.abs(peak_index) + 1 >= len(peak_times):
            break

        mean_time = peak_times[peak_index] + (peak_times[peak_index] - peak_times[peak_index + 1]) / 2

        end_point_1_2d = [s1[peaks[peak_index]], s2[peaks[peak_index]]]
        end_point_2_2d = [s1[peaks[peak_index + 1]], s2[peaks[peak_index + 1]]]

        v = None

        if height_axis == 'x':
            end_point_1 = np.array([0, end_point_1_2d[0], end_point_1_2d[1]])
            end_point_2 = np.array([0, end_point_2_2d[0], end_point_2_2d[1]])

            v = end_point_1 - end_point_2

        elif height_axis == 'y':
            end_point_1 = np.array([end_point_1_2d[0], 0, end_point_1_2d[1]])
            end_point_2 = np.array([end_point_2_2d[0], 0, end_point_2_2d[1]])

            v = end_point_1 - end_point_2

        elif height_axis == 'z':
            end_point_1 = np.array([end_point_1_2d[0], end_point_1_2d[1], 0])
            end_point_2 = np.array([end_point_2_2d[0], end_point_2_2d[1], 0])

            v = end_point_1 - end_point_2

        nh = None

        if height_axis == 'x':
            nh = [1, 0, 0]

        elif height_axis == 'y':
            nh = [0, 1, 0]

        elif height_axis == 'z':
            nh = [0, 0, 1]

        n = np.cross(v, nh)

        normal_vectors.append(n)
        timestamps.append(mean_time)

    return normal_vectors, timestamps

def get_amplitude(x, y):
    ox = np.mean(x)
    oy = np.mean(y)

    origin = np.array([ox, oy])[:, np.newaxis]

    return np.linalg.norm([x, y] - origin, axis=0)

def project_swing_on_floor(x, y, t, swing_indices):
    amplitude = get_amplitude(x, y)

    peaks, _ = sp.signal.find_peaks(amplitude)

    vectors = []
    timestamps = []

    for i in swing_indices:
        i = i*2

        if i + 1 >= len(peaks):
            break

        x1 = x[peaks[i]]
        y1 = y[peaks[i]]

        x2 = x[peaks[i + 1]]
        y2 = y[peaks[i + 1]]

        v = [x1 - x2, y1 - y2]

        vectors.append(v)

        timestamp = (t[peaks[i]] + t[peaks[i + 1]]) / 2

        timestamps.append(timestamp)

    return np.array(vectors), np.array(timestamps)


def normalize_vectors(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    
def angle_between_vectors(v1, v2):
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def sanitize_vectors(vectors, timestamps):
    tolerance = 0.5

    last = None
    orthogonal_indices = []

    for index, vector in enumerate(vectors):
        if last is None:
            last = vector
            continue

        dot = np.dot(vector, last)
        
        if np.abs(dot) < tolerance:
            orthogonal_indices.append(index)
        else:
            last = vector

    vectors = np.delete(vectors, orthogonal_indices, axis=0)
    timestamps = np.delete(timestamps, orthogonal_indices)

    pivot_indices = []

    tolerance = 0.1

    last = None

    for index, vector in enumerate(vectors):
        if last is None:
            last = vector
            continue

        x1 = vector[0]
        y1 = vector[1]

        x2 = last[0]
        y2 = last[1]

        if np.abs(x1 - x2) > tolerance or np.abs(y1 - y2) > tolerance:
            pivot_indices.append(index)
            last = -vector

        else:
            last = vector

    vectors[pivot_indices] *= -1

    return vectors, timestamps
