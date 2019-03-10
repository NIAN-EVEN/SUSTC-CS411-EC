import matplotlib.pyplot as plt
import numpy as np

def save_data(filename, record):
    with open(filename+".csv", 'a') as f:
        for i, s in enumerate(record):
            f.write(str(i)+", "+str(s)+"\n")

def figure_plot(figurename, record):
    record_num = len(record)
    x = np.zeros(record_num)
    y = np.zeros(record_num)
    for i, s in enumerate(record):
        x[i] = i
        y[i] = s
    plt.figure()
    plt.plot(x, y)
    plt.title(figurename)
    plt.xlabel("generation")
    plt.ylabel("function value")
    plt.savefig(figurename+ ".png")

