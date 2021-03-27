import random
import numpy as np
import matplotlib.pyplot as plt

def rand_seed(m, b, num=2):
    # create empty list
    x_coor = []
    y_coor = []
    label = []
    # positive and negtive point number
    pos_num = int(num / 2)
    neg_num = num - pos_num
    # random create point
    for i in range(pos_num):
        x = random.uniform(0, 30)
        r = random.uniform(1, 30)
        y = m * x + b - r
        # save the coordinate of x and y
        x_coor.append(x)
        y_coor.append(y)
        # save label, right=1, left=0
        label.append(1 if m >= 0 else -1)

    for i in range(neg_num):
        x = random.uniform(0, 30)
        r = random.uniform(1, 30)
        y = m * x + b + r
        x_coor.append(x)
        y_coor.append(y)
        label.append(-1 if m >= 0 else 1)
    return x_coor, y_coor, label

def get_sign(w, x):
    return np.sign(np.sum(w * x))

def PLA(x, y, w, num):
    
    iteration = 0
    while True:
        error_exist = False
        rand_list = random.sample(range(num), num)
        for i in rand_list:
            if get_sign(w, x[i, :].reshape([1, 3])) != y[i]:
                w = w + y[i] * x[i, :].reshape([1, 3])
                error_exist = True
                iteration += 1
                break

        if error_exist == False:
            break;

    return w, iteration

def Pocket(x, y, w, num, threshold, continuous_threshold):
    iteration = 0
    count = 0
    while iteration < threshold and count < continuous_threshold:
        i = random.randint(0, num - 1)

        if get_sign(w, x[i, :].reshape([1, 3])) != y[i]:
            wt = w + y[i] * x[i, :].reshape([1, 3])
            
            error_rate = verification(x, y, w, num, iteration=None, show=False)
            error_rate_t = verification(x, y, wt, num, iteration=None, show=False)

            if error_rate_t < error_rate:
                w = wt
                count = 0
            else:
                count += 1

            iteration += 1

            if error_rate_t == 0:
                break

    return w, iteration

def build_data(m, b, num):
    # x = 2D data, (1, x0, x1) = (1, x, y)
    x = np.zeros([num, 3])
    
    # label of data
    y = np.zeros([num])

    # build 2D data samples
    x[:, 1], x[:, 2], y = rand_seed(m, b, num)
    x[:, 0] = 1

    return x, y

def verification(x, y, w, num, iteration, show=False):
    error_count = 0

    for i in range(num):
        if get_sign(w, x[i, :].reshape([1, 3])) != y[i]:
            error_count += 1

    if show == True:
        print(f"y = {-w[0, 1] / w[0, 2]:.5f} * x + {-w[0, 0] / w[0, 2]:.5f}")
        print(f"iteration = {iteration}")
        if error_count == 0:
            print("Correct Line Equation!!!")
        else:
            print("Wrong Line Equation!!!")

    return error_count/num

class plt_proc():

    def __init__(self, x, num, title):
        # set the figure
        plt.figure(figsize=(13, 11))
        plt.title(title)
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")

        # plot the data
        plt.plot(x[:int(num/2), 1], x[:int(num/2), 2], 'o', label="data with label = 1")
        plt.plot(x[int(num/2):, 1], x[int(num/2):, 2], 'x', label="data with label = -1")

    def add_line(self, w, m, b, num, iteration, label, txt):
        # plot the line equation
        # get m and b from w
        if w is not None:
            m = -1 * w[0, 1] / w[0, 2]
            b = -1 * w[0, 0] / w[0, 2]
        
        x = np.arange(30 + 1)
        y = m * x + b

        plt.plot(x, y, label=f"{label}: y = {m:.5f} * x + {b:.5f}{txt}")

    def save_and_show(self, itr_avg, filename, avg_show=False):
        if avg_show is True:
            plt.title(f"Avg. Iteration = {itr_avg:.3f}", loc="right")

        # set the legend of the figure
        plt.legend(loc="best")

        # save the figure
        plt.savefig(filename, dpi=800, bbox_inches="tight")

        # show the figure
        plt.show()