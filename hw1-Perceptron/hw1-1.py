import numpy as np

from utils import (
    PLA, build_data,
    verification, plt_proc
)

def PLA_3_times_with_30_data(m, b, num, times):
    # build 2D data
    x, y = build_data(m, b, num)

    #initial weight w = (0, 0, 0)
    w = np.zeros([1,3])

    # count the iteration total numbers
    iteration_count = 0

    # initial the figure
    plt_fig = plt_proc(x, num, title="HW1-1: PLA with 30 2D data samples")

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

    for i in range(times):

        # run PLA algorithm
        w_result, iteration = PLA(x, y, w, num)

        # verify the line equation
        verification(x, y, w_result, num, iteration, show=True)

        # count the total number of iterations
        iteration_count += iteration

        # plot the line equation
        plt_fig.add_line(
            w=w_result,
            m=None,
            b=None,
            num=num,
            iteration=iteration,
            label=f"{i}",
            txt=f", iteration = {iteration}"
        )

    print(f"Avg. Iteration = {iteration_count/3:.3f}")

    # save and show the figure
    plt_fig.save_and_show(
        itr_avg=iteration_count / 3,
        filename='hw1-1.png',
        avg_show=True
    )

if __name__ == '__main__':
    
    # y = m * x + b
    m = 1
    b = 2
    
    # default number of data = 30
    num = 30
    
    # set the running times of PLA
    times = 3

    PLA_3_times_with_30_data(m, b, num, times)