import sys

import matplotlib.pyplot as plt

from reversion import get_all_periods

def sign(num):
    return -1 if num < 0 else 1

if __name__ == '__main__':
    args = sys.argv[1:]

    series = args[0]

    periods_a, periods_b = get_all_periods(series, 'y')

    last = None
    
    ka, ma = None, None
    kb, mb = None, None

    for period_a, period_b in zip(periods_a, periods_b):
        if last is None:
            last = {'a': period_a, 'b': period_b}
            continue

        if sign(last['a']['period'] - last['b']['period']) != sign(period_a['period'] - period_b['period']):
            ka = (period_a['period'] - last['a']['period']) / (period_a['distance'] - last['a']['distance'])
            kb = (period_b['period'] - last['b']['period']) / (period_b['distance'] - last['b']['distance'])

            ma = last['a']['period'] - ka * last['a']['distance']
            mb = last['b']['period'] - kb * last['b']['distance']

            break

        last = {'a': period_a, 'b': period_b}

    d_intersect = (mb - ma) / (ka - kb)
    t_intersect = ka * d_intersect + ma

    print("Intersect at: " + str(d_intersect))

    # ta = [a['period'] for a in periods_a]
    # da = [a['distance'] for a in periods_a]

    # tb = [a['period'] for a in periods_b]
    # db = [a['distance'] for a in periods_b]

    # plt.plot(da, ta, 'o-', label='A')
    # plt.plot(db, tb, 'o-', label='B')

    # plt.vlines([d_intersect], min(min(ta), min(tb)), max(max(ta), max(tb)))

    # plt.legend()

    # plt.show()
        