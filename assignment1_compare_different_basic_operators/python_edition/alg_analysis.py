import numpy as np
import matplotlib.pyplot as plt
import os, random, copy, math, re
import pandas as pd

def save_data(filename, record):
    with open(filename+".csv", 'a') as f:
        for i, s in enumerate(record):
            f.write(str(i)+", "+str(s)+"\n")

def transform_line(line):
    if line[0] == '[':
        return None
    else:
        g, f = line.strip().split(',')
        return int(g.split('=')[1]), float(f.split('=')[1])

def evolve_process(file, generation_size, run):
    generation = np.zeros(generation_size)
    fitness = np.zeros(generation_size)
    with open(file, 'r') as f:
        pos = 0
        for line in f.readlines():
            result = transform_line(line)
            if result == None:
                pos = 0
            else:
                generation[pos] += result[0]
                fitness[pos] += result[1]
                pos += 1
    return generation/run, fitness/run


def show_all(DIR):
    files = os.listdir(DIR)
    generation_size = {"ackley": 101, "gold_stein_price": 7, "quartic_noise": 201,
                       "rastrigin": 334, "shekel": 7, "step": 101}
    show_data = ['023','013','012']
    for file in files:
        if file[0] in show_data[0] and file[1] in show_data[1] and file[2] in show_data[2]:
            plt.figure()
            func = file[4:-13]
            generation, fitness = evolve_process(DIR+file, generation_size[func], 10)
            plt.plot(generation, fitness)
            plt.title(file[:-13])
            plt.xlabel("generation")
            plt.ylabel("fitness")
            plt.yscale("log")
            plt.text(generation[-1], fitness[-1], '%.2f'%fitness[-1])
            plt.savefig("../graph/%s.png"%file[:-13])
            # plt.show()


def performance_vector(file, generation_size, run):
    best_record = []
    with open(file, 'r') as f:
        pos = 0
        for line in f.readlines():
            result = transform_line(line)
            if result == None:
                pos = 0
            else:
                if pos == generation_size - 1:
                    best_record.append(result[1])
                pos += 1
    if len(best_record) == 0:
        return math.nan, math.nan, math.nan
    best_record = np.array(best_record)
    return best_record.min(), best_record.mean(), best_record.std()

def performance_table(DIR):
    min_table = {"function":[], "ackley":[], "gold_stein_price":[], "quartic_noise":[],
                 "rastrigin":[], "shekel":[], "step":[]}
    mean_table = {"function":[], "ackley":[], "gold_stein_price":[], "quartic_noise":[],
                 "rastrigin":[], "shekel":[], "step":[]}
    std_table = {"function":[], "ackley":[], "gold_stein_price":[], "quartic_noise":[],
                 "rastrigin":[], "shekel":[], "step":[]}
    generation_size = {"ackley":101, "gold_stein_price":7, "quartic_noise":201,
                 "rastrigin":334, "shekel":7, "step":101}
    files = os.listdir(DIR)
    pre_function = "__"
    for file in files:
        part = re.match(FILE_NAME_FORMAT, file)
        func = part.group(2)
        vec = performance_vector(DIR+file, generation_size[func], 10)
        if pre_function != part.group(1):
            min_table["function"].append(part.group(1))
            mean_table["function"].append(part.group(1))
            std_table["function"].append(part.group(1))
            pre_function = part.group(1)
        min_table[func].append(vec[0])
        mean_table[func].append(vec[1])
        std_table[func].append(vec[2])
    # return min_table, mean_table, std_table
    return pd.DataFrame(min_table), pd.DataFrame(mean_table), \
               pd.DataFrame(std_table)

if __name__ == "__main__":
    print(os.getcwd())
    DIR = "..\\data\\EP\\"
    FILE_NAME_FORMAT = r'^(\d{2,3})\_([a-z\_]+)\_function\.txt'
    min_table, mean_table, std_table = performance_table(DIR)
    # show_all(DIR)








