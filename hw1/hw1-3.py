import numpy as np
from time import time
import os
from utils import (
    PLA, Pocket, plt_proc, build_data,
    rand_seed, get_sign, verification, 
)

def pocket_vs_pla(m, b, num, threshold, continuous_threshold):
    # build 2D data
    x, y = build_data(m, b, num)

    #initial weight w = (0, 0, 0)
    w = np.zeros([1,3])

    # initial the figure
    plt_fig = plt_proc(x, num, title="HW1-3: Pocket v.s. PLA with 1000 2D data samples")

    # plot the sample line equation
    plt_fig.add_line(
        w=None,
        m=m,
        b=b,
        num=num,
        iteration=None,
        label=f"Benchmark",
        txt=""
    )

    # set the start time of PLA algorithm
    PLA_start = time()

    # run PLA algorithm
    w_PLA, iteration_PLA = PLA(x, y, w, num)

    # get the execution time of PLA
    PLA_exe_time = time() - PLA_start

    PLA_error_rate = verification(x, y, w_PLA, num, iteration=None, show=False)
    
    # plot the PLA line equation
    plt_fig.add_line(
        w=w_PLA,
        m=None,
        b=None,
        num=num,
        iteration=iteration_PLA,
        label=f"PLA",
        txt=f"\n        error rate = {PLA_error_rate:.03f}, iteration = {iteration_PLA}, exec. time = {PLA_exe_time:.03f}"
    )

    # set the start time of PLA algorithm
    Pocket_start = time()

    # run Pocket algorithm
    w_Pocket, iteration_Pocket = Pocket(x, y, w, num, threshold, continuous_threshold)

    # get the execution time of PLA
    Pocket_exe_time = time() - Pocket_start

    Pocket_error_rate = verification(x, y, w_Pocket, num, iteration_Pocket, show=False)

    # plot the Pocket line equation
    plt_fig.add_line(
        w=w_Pocket,
        m=None,
        b=None,
        num=num,
        iteration=iteration_Pocket,
        label=f"Pocket",
        txt=f"\n        error rate = {Pocket_error_rate:.03f}, iteration = {iteration_Pocket}, exec. time = {Pocket_exe_time:.03f}"
    )

    print(f"PLA execution time = {PLA_exe_time:.5f} seconds")
    print(f"PLA Iteration = {iteration_PLA}")
    print(f"PLA error rate = {PLA_error_rate}\n")

    print(f"Pocket execution time = {Pocket_exe_time:.5f} seconds")
    print(f"Pocket Iteration = {iteration_Pocket}")
    print(f"Pocket error rate = {Pocket_error_rate:.03f}")

    # save and show the figure
    plt_fig.save_and_show(
        itr_avg=None,
        filename='hw1-3.png',
        avg_show=False
    )

if __name__ == '__main__':
    
    # y = m * x + b
    m = 1
    b = 2
    
    # default number of data = 30
    num = 1000

    # set the threshold
    threshold = 10000
    continuous_threshold = 100

    pocket_vs_pla(m, b, num, threshold, continuous_threshold)