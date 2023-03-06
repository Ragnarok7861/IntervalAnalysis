from csv import reader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Reading data from files ----------------------------------------------------------------------------------------------

Ch1_U_mV = []
Ch2_U_mV = []

with open('Канал 1_700nm_0.23mm.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    for row in csv_reader:
        row = row[0].split(';')
        Ch1_U_mV.append(row[0])

with open('Канал 2_700nm_0.23mm.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    for row in csv_reader:
        row = row[0].split(';')
        Ch2_U_mV.append(row[0])

Ch1_U_mV.remove('мВ')
Ch2_U_mV.remove('мВ')

Ch1_U = np.empty(len(Ch1_U_mV), dtype=float)
Ch2_U = np.empty(len(Ch2_U_mV), dtype=float)
ind = np.empty(len(Ch1_U_mV), dtype=int)
regression1 = np.empty(len(Ch1_U_mV), dtype=float)
regression2 = np.empty(len(Ch1_U_mV), dtype=float)

for i in range(0, len(Ch1_U_mV)):
    Ch1_U[i] = float(Ch1_U_mV[i])
    Ch2_U[i] = float(Ch2_U_mV[i])

for i in range(0, 200):
    ind[i] = i + 1

# Plot data (if needed) ------------------------------------------------------------------------------------------------

PlotDataIsNeeded = True

if PlotDataIsNeeded:
    fig, ax = plt.subplots()
    ax.plot(ind, Ch1_U, label="Channel 1", color="blue")
    ax.plot(ind, Ch2_U, label="Channel 2", color="orange")
    ax.set_xlabel('n')
    ax.set_ylabel('mV')
    plt.xticks(np.arange(0, 201, 20))
    ax.legend()
    # ax.grid()
    plt.title("Data from experiment")
    fig.show()

    fig, ax = plt.subplots()
    ax.plot(ind, Ch1_U, label="Channel 1", color="blue")
    ax.set_xlabel('n')
    ax.set_ylabel('mV')
    plt.xticks(np.arange(0, 201, 20))
    ax.legend()
    # ax.grid()
    plt.title("Data from channel 1")
    fig.show()

    fig, ax = plt.subplots()
    ax.plot(ind, Ch2_U, label="Channel 2", color="orange")
    ax.set_xlabel('n')
    ax.set_ylabel('mV')
    plt.xticks(np.arange(0, 201, 20))
    ax.legend()
    # ax.grid()
    plt.title("Data from channel 2")
    fig.show()

    plt.figure()

# Building linear regression -------------------------------------------------------------------------------------------

# cropping bad values for regression building
CROP_X_1 = 20
CROP_Y_1 = 181

arr2 = Ch1_U[CROP_X_1:CROP_Y_1]
arr1 = np.array(ind[CROP_X_1:CROP_Y_1]).reshape((-1, 1))

# coefficients calculation
model = LinearRegression().fit(arr1, arr2)
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")

A = model.coef_
B = model.intercept_

for i in range(0, len(ind)):
    regression1[i] = A*ind[i] + B

# cropping bad values for regression building
CROP_X_2 = 20
CROP_Y_2 = 181

arr4 = Ch2_U[CROP_X_2:CROP_Y_2]
arr3 = np.array(ind[CROP_X_2:CROP_Y_2]).reshape((-1, 1))

model2 = LinearRegression().fit(arr3, arr4)
print(f"intercept: {model2.intercept_}")
print(f"slope: {model2.coef_}")

# coefficients calculation
A2 = model2.coef_
B2 = model2.intercept_

for i in range(0, len(ind)):
    regression2[i] = A2*ind[i] + B2

# Calculate intervals --------------------------------------------------------------------------------------------------

# multiplier to extern interval for regression capture
ConstBroaden = 0.00005

err1 = np.empty(len(Ch1_U), dtype=float)
for i in range(0, len(Ch1_U)):
    err1[i] = abs(Ch1_U[i] - regression1[i])

err2 = np.empty(len(Ch2_U), dtype=float)
for i in range(0, len(Ch2_U)):
    err2[i] = abs(Ch2_U[i] - regression2[i])

X1_inter = np.empty(shape=(len(Ch1_U), 2), dtype='float')
X1_inter_d = np.empty(shape=(len(Ch1_U), 2), dtype='float')
X2_inter = np.empty(shape=(len(Ch2_U), 2), dtype='float')
X2_inter_d = np.empty(shape=(len(Ch2_U), 2), dtype='float')

# making interval array and subtracting linear dependency to make constant
for i in range(len(Ch1_U)):
    X1_inter[i][0] = Ch1_U[i] - err1[i] - ConstBroaden
    X1_inter[i][1] = Ch1_U[i] + err1[i] + ConstBroaden
    X1_inter_d[i][0] = X1_inter[i][0] - A * ind[i]
    X1_inter_d[i][1] = X1_inter[i][1] - A * ind[i]

X2_inter = np.empty(shape=(len(Ch1_U), 2), dtype='float')
for i in range(len(Ch2_U)):
    X2_inter[i][0] = Ch2_U[i] - err2[i] - ConstBroaden
    X2_inter[i][1] = Ch2_U[i] + err2[i] + ConstBroaden
    X2_inter_d[i][0] = X2_inter[i][0] - A2 * ind[i]
    X2_inter_d[i][1] = X2_inter[i][1] - A2 * ind[i]

# arrays just for plotting
data_d1 = np.empty(shape=len(Ch1_U), dtype='float')
data_err1 = np.empty(shape=len(Ch1_U), dtype='float')
for i in range(len(Ch1_U)):
    data_d1[i] = 0.5 * (X1_inter_d[i][0] + X1_inter_d[i][1])
    data_err1[i] = 0.5 * (X1_inter_d[i][1] - X1_inter_d[i][0])

data_d2 = np.empty(shape=len(Ch2_U), dtype='float')
data_err2 = np.empty(shape=len(Ch2_U), dtype='float')
for i in range(len(Ch2_U)):
    data_d2[i] = 0.5 * (X2_inter_d[i][0] + X2_inter_d[i][1])
    data_err2[i] = 0.5 * (X2_inter_d[i][1] - X2_inter_d[i][0])

# R external estimation

maxd1 = max(X1_inter_d[0][0] / X2_inter_d[0][0], X1_inter_d[0][0] / X2_inter_d[0][1], X1_inter_d[0][1] / X2_inter_d[0][0], X1_inter_d[0][1] / X2_inter_d[0][1])
mind1 = min(X1_inter_d[0][0] / X2_inter_d[0][0], X1_inter_d[0][0] / X2_inter_d[0][1], X1_inter_d[0][1] / X2_inter_d[0][0], X1_inter_d[0][1] / X2_inter_d[0][1])
for i in range(1, len(data_d1), 1):
    maxd1 = max(maxd1, X1_inter_d[i][0] / X2_inter_d[i][0], X1_inter_d[i][0] / X2_inter_d[i][1], X1_inter_d[i][1] / X2_inter_d[i][0], X1_inter_d[i][1] / X2_inter_d[i][1])
    mind1 = min(maxd1, X1_inter_d[i][0] / X2_inter_d[i][0], X1_inter_d[i][0] / X2_inter_d[i][1], X1_inter_d[i][1] / X2_inter_d[i][0], X1_inter_d[i][1] / X2_inter_d[i][1])
print("Rext1 = ", mind1)
print("Rext2 = ", maxd1)

# Calculating Jaccard multimetric and R inner estimation ---------------------------------------------------------------
stepnumb = 200
step = (maxd1 - mind1)/stepnumb
print("step =", step)
R_interval = [step * i + mind1 for i in range(stepnumb)]
Jaccars = []

for R in R_interval:
    all_intervals = np.concatenate((X1_inter_d, R * X2_inter_d), axis=0)
    intersection = all_intervals[0]
    union = all_intervals[0]
    for i in range(1, len(all_intervals), 1):
        intersection = [max(intersection[0], all_intervals[i][0]), min(intersection[1], all_intervals[i][1])]
        union = [min(union[0], all_intervals[i][0]), max(union[1], all_intervals[i][1])]
    JK = (intersection[1] - intersection[0])/(union[1] - union[0])
    Jaccars.append(JK)
    print("JK =", JK, "R =", R)
    del all_intervals
print('MAX Jaccard =', max(Jaccars))


# Plot Jaccard metric from R (write correct values) --------------------------------------------------------------------

plt.plot(R_interval, Jaccars, label="Jaccard metric", zorder=1)
plt.scatter(0.9150431449664217, 0.08833511445728766, label="optimal point at R=" + str(0.91504), color="b")
plt.scatter(0.9138541648703038, 0.003250759965497126, label="Rmin=" + str(0.91385), color="r", zorder=2)
plt.scatter(0.9162321250625396, 0.002060077331739088, label="Rmax=" + str(0.91623), color="r", zorder=2)
plt.legend()
plt.xlabel('R')
plt.ylabel('Jaccard')
plt.title('Jaccard metric')

# Plot intervals after linear dependency subtraction and finding optimal R ---------------------------------------------

Roptimal = 0.91504

fig, ax = plt.subplots()
# ax.plot(ind, Ch1_U, label="Channel 1 data", color="blue")
ax.errorbar(ind, data_d1, yerr=data_err1, label="Interval data X1\'", marker='none', linestyle='none', elinewidth=0.8, capsize=4, capthick=1)
ax.errorbar(ind, Roptimal * data_d2, yerr=data_err2, label="Interval data R*X2\'", marker='none', linestyle='none', elinewidth=0.8, capsize=4, capthick=1)
# ax.errorbar(ind, Ch1_U, yerr=err1, label="Interval data", marker='none', linestyle='none', color="pink")
ax.set_xlabel('n')
ax.set_ylabel('X\'')
plt.xticks(np.arange(0, 201, 20))
ax.legend()
# ax.grid()
plt.title("X1\' and R*X2\' intervals")
fig.show()

# Plot linear regression -----------------------------------------------------------------------------------------------

PlotRegressionIsNeeded = True


if PlotRegressionIsNeeded:
    fig, ax = plt.subplots()
    ax.plot(ind, Ch1_U, label="Channel 1 data", color="blue")
    ax.plot(ind, regression1, label="Regression 1", color="red")
    ax.errorbar(ind, Ch1_U, yerr=err1+ConstBroaden, label="Interval data", marker='none', linestyle='none')
    ax.set_xlabel('n')
    ax.set_ylabel('mV')
    plt.xticks(np.arange(0, 201, 20))
    ax.legend()
    # ax.grid()
    plt.title("Channel 1 Linear regression")
    fig.show()

    fig, ax = plt.subplots()
    ax.plot(ind, Ch2_U, label="Channel 2 data", color="orange")
    ax.plot(ind, regression2, label="Regression 2", color="green")
    ax.errorbar(ind, Ch2_U, yerr=err2+ConstBroaden, label="Interval data", marker='none', linestyle='none', color="pink")
    ax.set_xlabel('n')
    ax.set_ylabel('mV')
    plt.xticks(np.arange(0, 201, 20))
    ax.legend()
    # ax.grid()
    plt.title("Channel 2 Linear regression")
    fig.show()


plt.show()
