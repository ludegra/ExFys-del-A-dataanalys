import sys

import matplotlib.pyplot as plt

from foucault import get_dataset

if __name__ == '__main__':
    args = sys.argv[1:]

    series = args[0]
    name = args[1]

    df = get_dataset(series, name)

    fig = plt.figure(figsize=(15, 7))

    ax = fig.subplots(3, 1)

    ax[0].plot(df['t'], df['x'], label='x')
    ax[0].set_title("x with respect to time")
    ax[0].set_xlabel('t')
    ax[0].set_ylabel('x')

    ax[1].plot(df['t'], df['y'], label='y')
    ax[1].set_title("y with respect to time")
    ax[1].set_xlabel('t')
    ax[1].set_ylabel('y')

    ax[2].plot(df['t'], df['z'], label='z')
    ax[2].set_title("z with respect to time")
    ax[2].set_xlabel('t')
    ax[2].set_ylabel('z')

    plt.show()

