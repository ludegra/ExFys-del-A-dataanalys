import sys

from reversion import get_dataset, get_mean_period


if __name__ == '__main__':
    args = sys.argv[1:]

    series = args[0]
    pivot = args[1]
    distance = args[2]

    df = get_dataset(series, pivot, distance)

    period = get_mean_period(df, 'y')

    print("Period: " + str(period))


