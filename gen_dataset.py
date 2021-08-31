import csv
import numpy as np
import datetime
from scipy.stats import skewnorm


def naming_dir():
    now = str(datetime.datetime.now())
    ymd = "".join(now.split(' ')[0].split('-'))[2:]
    hms = "".join(now.split(' ')[1].split(':')[0:2]) + now.split('.')[0].split(':')[-1]
    return ymd+'_'+hms


def make_normal(no):
    mu = np.random.randint(-int(no) / 100, int(no) / 100)
    std = np.random.randint(1, int(no) / 100)
    skew = np.random.randint(-int(no) / 100, int(no) / 100)
    xrange = np.linspace(mu - std*10, mu + std*10, no)
    pdf = skewnorm.pdf(xrange, skew, mu, std)
    normal = np.random.choice(xrange, no, p=pdf/np.sum(pdf))
    el = np.zeros(1)
    i = np.random.randint(1, 100)
    # spike
    if i < 20:
        el[0] = 1
        no_spike = np.random.randint(1, int(no / 10))
        loc_spike = [np.random.randint(no) for ii in range(no_spike)]
        for i in loc_spike:
            normal[loc_spike] = normal[loc_spike] * np.random.randint(-5, 5)
        return np.concatenate((el, normal))

    # peak change
    if 21 <= i < 40:
        el[0] = 2
        loc_peak_change = np.random.randint(no)
        new_peak = mu * np.random.randint(-5, 5)
        normal[loc_peak_change:] = np.random.normal(new_peak, std, no - loc_peak_change)
        return np.concatenate((el, normal))

    else:
        return np.concatenate((el, normal))


def make_drift(no):
    X = np.linspace(0, no - 1, no)
    slope = np.random.randint(int(no / 10), int(no / 5))
    bias = np.random.randint(int(no / 10), int(no / 5))
    drift = slope * X + bias
    loc_drop = np.random.randint(0, no)
    length_drop = np.random.randint(int(no / 10), int(no / 2))
    bias_drop = np.random.randint(int(min(drift)), int(max(drift)))
    i = np.random.randint(1, 100)
    el = np.zeros(1)

    # randomly drop twice, slope changes
    if i < 10:
        el[0] = 2
        init = 0
        slope = 1
        end = init + no * slope

        jump_size = np.random.randint(0, end)
        change_point = np.random.randint(0, end)

        base = np.arange(init, end, slope, dtype="float64")
        x = np.arange(0, no, 1)

        reset_point = np.random.randint(init, end)
        base[reset_point:] -= base[reset_point]

        jump_point = np.random.randint(init, end)
        base[jump_point:] += jump_size
        slope_change = np.random.randint(1, 4)
        base[change_point:] *= slope_change
        base[change_point:] -= base[change_point] / slope_change

        noise = np.random.rand(np.size(base)) * 2
        base = base + noise

        return np.concatenate((el, np.array(base)))

    # randomly drop once
    if 10 <= i < 25:
        el[0] = 1
        init = 0
        slope = 1
        end = init + no * slope

        jump_size = np.random.randint(0, end)

        base = np.arange(init, end, slope, dtype="float64")

        reset_point = np.random.randint(init, end)
        base[reset_point:] -= base[reset_point]

        jump_point = np.random.randint(init, end)
        base[jump_point:] += jump_size

        noise = np.random.rand(np.size(base)) * 2
        base = base + noise

        return np.concatenate((el, np.array(base)))

    else:
        print(drift)
        print(type(drift))
        return np.concatenate((el, np.array(drift)))


def make_spike(no):
    X = np.zeros(no)
    i = np.random.randint(1, 100)
    no_spike = np.random.randint(1, int(no / 10))
    loc_spike = [np.random.randint(no) for i in range(no_spike)]
    el = np.zeros(1)
    # spike has one level
    if i < 75:
        el[0] = 0
        for ii in range(no_spike):
            spike_noise = float(np.random.random(1) * 0.01)
            X[loc_spike[ii]] = 1 + spike_noise
        return np.concatenate((el, np.array(X)))

    if 75 <= i < 90:
        el[0] = 1
        level_spike = np.concatenate([np.ones(70), 2 * np.ones(30)])
        for ii in range(50):
            for iii in range(no_spike):
                X[loc_spike[iii]] = np.random.choice(level_spike) + float(np.random.random(1) * 0.01)
        return np.concatenate((el, np.array(X)))

    if 90 <= i:
        el[0] = 2
        level_spike = np.concatenate([np.ones(50), 2 * np.ones(30), 3 * np.ones(20), 4 * np.ones(10)])
        for ii in range(50):
            for iii in range(no_spike):
                X[loc_spike[iii]] = np.random.choice(level_spike) + float(np.random.random(1) * 0.01)
        return np.concatenate((el, np.array(X)))


def save_raw_data(no_data, iteration, csv_name):
    f = open(f'./{csv_name}_raw.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    data = np.array([])
    for i in range(iteration):
        label_no = np.random.randint(3)
        print(f'iteration: {i}')
        print(f'label: {label_no}')
        if label_no == 0:  # normal
            data = np.concatenate((np.array([label_no]), make_normal(no_data)))
        elif label_no == 1:  # drift
            data = np.concatenate((np.array([label_no]), make_drift(no_data)))
        elif label_no == 2:  # spike
            data = np.concatenate((np.array([label_no]), make_spike(no_data)))

        wr.writerow(data)


if __name__ == "__main__":

    csv_name = f'{naming_dir()}'
    no_data = 1000
    iteration = 100

    save_raw_data(no_data=no_data, iteration=iteration, csv_name=csv_name)
