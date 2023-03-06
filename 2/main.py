from csv import DictReader, reader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

numValues = 200

def readDataFromFile(file1='Канал 1_700nm_0.05.csv', file2='Канал 2_700nm_0.05.csv'):
    ch1U = []
    ch2U = []

    with open(file1, 'r') as f:
        csv_reader = DictReader(f, delimiter=';', )
        for row in csv_reader:
            ch1U.append(float(row['x']))  # actually row[0]

    with open(file2, 'r') as f:
        csv_reader = DictReader(f, delimiter=';')
        for row in csv_reader:
            ch2U.append(float(row['x']))  # actually row[0]

    ch1U = np.asarray(ch1U, dtype=float)
    ch2U = np.asarray(ch2U, dtype=float)

    return ch1U, ch2U

def readWeightsDataFromFile(file='w.csv'):
    w1 = []
    w2 = []

    with open(file, 'r') as f:
        csv_reader = DictReader(f, delimiter=';', )
        for row in csv_reader:
            w1.append(float(row['w1']))  # actually row[0]
            w2.append(float(row['w2']))  # actually row[1]

    w1 = np.asarray(w1, dtype=float)
    w2 = np.asarray(w2, dtype=float)

    return w1, w2

def plotData(X, Y, legends, colors, xylabels, title, show=True):
    fig, ax = plt.subplots()
    for x, y, legend, color, xylabel in zip(X, Y, legends, colors, xylabels):
        ax.plot(x, y, label=legend, color=color)
        ax.set_xlabel(xylabel[0])
        ax.set_ylabel(xylabel[1])
    ax.legend()
    plt.title(title)
    if show:
        fig.show()
    return fig, ax


def buildLinearRegression(channelU):
    # cropping bad values for regression building
    start = 15
    end = 195

    arr2 = channelU[start:end]
    arr1 = np.arange(start, end).reshape((-1, 1))

    # coefficients calculation
    model = LinearRegression().fit(arr1, arr2)
    print(f"intercept: {model.intercept_}")
    print(f"slope: {model.coef_}")

    a = model.coef_
    b = model.intercept_

    regression = a * np.arange(0, numValues) + b
    return regression, a


def makeIntervals(channelU, regression, a):
    # multiplier to extern interval for regression capture
    tolerance = np.empty(shape=(len(channelU),), dtype=float)
    tolerance.fill(5e-5)

    err = np.abs(channelU - regression)
    print('error linear =', err)
    X_inter = np.empty(shape=(len(channelU), 2), dtype='float')
    X_inter_d = np.empty(shape=(len(channelU), 2), dtype='float')

    ind = np.arange(0, numValues)

    # making interval array and subtracting linear dependency to make constant
    X_inter[:, 0] = channelU - err - tolerance
    X_inter[:, 1] = channelU + err + tolerance
    X_inter_d[:, 0] = X_inter[:, 0] - a * ind
    X_inter_d[:, 1] = X_inter[:, 1] - a * ind

    return X_inter_d

def makeIntervalsIntRegr(channelU, w, a):
    # multiplier to extern interval for regression capture
    tolerance = np.empty(shape=(numValues,), dtype=float)
    tolerance.fill(1e-4)

    err = w * tolerance
    constErr = 5e-5

    X_inter = np.empty(shape=(numValues, 2), dtype='float')
    X_inter_d = np.empty(shape=(numValues, 2), dtype='float')
    ind = np.arange(1, numValues+1)

    # making interval array and subtracting linear dependency to make constant
    X_inter[:, 0] = channelU - err - constErr
    X_inter[:, 1] = channelU + err + constErr

    for i in range(len(channelU)):
        X_inter_d[i][0] = X_inter[i][0] - max(a[0] * ind[i], a[1] * ind[i])
        X_inter_d[i][1] = X_inter[i][1] - min(a[0] * ind[i], a[1] * ind[i])

    return X_inter_d


# R external estimation
def externalEstimateR(X1_inter_d, X2_inter_d):
    maxd1 = max(X1_inter_d[0][0] / X2_inter_d[0][0], X1_inter_d[0][0] / X2_inter_d[0][1],
                X1_inter_d[0][1] / X2_inter_d[0][0], X1_inter_d[0][1] / X2_inter_d[0][1])
    mind1 = min(X1_inter_d[0][0] / X2_inter_d[0][0], X1_inter_d[0][0] / X2_inter_d[0][1],
                X1_inter_d[0][1] / X2_inter_d[0][0], X1_inter_d[0][1] / X2_inter_d[0][1])

    for i in range(1, len(X1_inter_d[0])):
        d1 = max(X1_inter_d[i][0] / X2_inter_d[i][0], X1_inter_d[i][0] / X2_inter_d[i][1],
                 X1_inter_d[i][1] / X2_inter_d[i][0], X1_inter_d[i][1] / X2_inter_d[i][1])
        maxd1 = max(maxd1, d1)
        d1 = min(X1_inter_d[i][0] / X2_inter_d[i][0], X1_inter_d[i][0] / X2_inter_d[i][1],
                 X1_inter_d[i][1] / X2_inter_d[i][0], X1_inter_d[i][1] / X2_inter_d[i][1])
        mind1 = min(mind1, d1)
    print("Rext1 = ", mind1)
    print("Rext2 = ", maxd1)
    return mind1, maxd1


def calcaulateJaccard(R, X1_inter_d, X2_inter_d):
    all_intervals = np.concatenate((X1_inter_d, R * X2_inter_d), axis=0)
    intersection = all_intervals[0]
    union = all_intervals[0]
    for i in range(1, len(all_intervals)):
        intersection = [max(intersection[0], all_intervals[i][0]), min(intersection[1], all_intervals[i][1])]
        union = [min(union[0], all_intervals[i][0]), max(union[1], all_intervals[i][1])]
    jc = (intersection[1] - intersection[0]) / (union[1] - union[0])
    return jc


def internalEstimateRJaccard(Rmin, Rmax, X1_inter_d, X2_inter_d):
    R_interval = np.linspace(Rmin - 0.06, Rmax + 0.06, 1500)
    Jaccars = []

    for R in R_interval:
        Jc = calcaulateJaccard(R, X1_inter_d, X2_inter_d)
        Jaccars.append(Jc)
        print('Jc =', Jc, 'R =', R)

    print('MAX Jaccard =', max(Jaccars))
    return R_interval, max(Jaccars), Jaccars, R_interval[np.argmax(Jaccars)]


def main():
    #    plt.rcParams['text.usetex'] = True
    U1, U2 = readDataFromFile()
    w1, w2 = readWeightsDataFromFile()
    num = np.arange(1, numValues+1)

    fig, ax = plotData(
        (num, num), (U1, U2), ('First channel', 'Second channel'),
        ('blue', 'orange'), (('n', 'mV'), ('n', 'mV')), 'Raw data')

    # b1 = [7.1411e-02, 7.1489e-02]
    a1 = [4.0914e-06, 4.5860e-06]

    # b2 = [7.8083e-02, 7.8176e-02]
    a2 = [2.9447e-06, 3.6127e-06]

    X1 = makeIntervalsIntRegr(U1, w1, a1)
    X2 = makeIntervalsIntRegr(U2, w2, a2)

    x1err = X1[:, 1] - X1[:, 0]
    x2err = X2[:, 1] - X2[:, 0]
    x1err_r = w1 * 1e-4
    x2err_r = w2 * 1e-4
    ax.errorbar(num, U1, yerr=x1err_r, color='blue', label='First channel')
    ax.errorbar(num, U2, yerr=x2err_r, color='orange', label='Second channel')
    fig.show()

    RminIn = 0.9125042911712944
    RmaxIn = 0.9165107552358179

    extRmin, extRmax = externalEstimateR(X1, X2)
    R_int, JaccardOpt, Jaccard, Ropt = internalEstimateRJaccard(extRmin, extRmax, X1, X2)
    fig, ax = plotData((R_int,),
                       (Jaccard,), ('Jaccard metric',), (None,),
                       (('$R_{21}$', 'Jaccard'),), 'Jaccard metric', show=False)
    ax.scatter(RminIn, calcaulateJaccard(RminIn, X1, X2), color='red', label=f'$R_{{min}}={RminIn:.5f}$', zorder=2)
    ax.scatter(RmaxIn, calcaulateJaccard(RmaxIn, X1, X2), color='red', label=f'$R_{{max}}={RmaxIn:.5f}$', zorder=2)
    ax.scatter(Ropt, JaccardOpt, color='blue', label=f'$R_{{opt}}={Ropt:.5f}$', zorder=2)
    ax.legend()
    fig.show()

    fig, ax = plt.subplots()
    # ax.plot(ind, Ch1_U, label="Channel 1 data", color="blue")
    ax.errorbar(num, (X1[:, 0] + X1[:, 1]) / 2, yerr=x1err, label="Interval data X1\'", elinewidth=0.8, capsize=4, capthick=1)
    ax.errorbar(num, Ropt * (X2[:, 0] + X2[:, 1]) / 2, yerr=x2err, label="Interval data R*X2\'", elinewidth=0.8, capsize=4, capthick=1)
    # ax.errorbar(ind, Ch1_U, yerr=err1, label="Interval data", marker='none', linestyle='none', color="pink")
    ax.set_xlabel('n')
    ax.set_ylabel('X\'')
    plt.xticks(np.arange(0, 201, 20))
    ax.legend()
    # ax.grid()
    plt.title("X1\' and R*X2\' intervals")
    fig.show()

    plt.show()

    print()


if __name__ == '__main__':
    main()
