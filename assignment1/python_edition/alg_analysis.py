import numpy as np
import matplotlib.pyplot as plt
import os, random, copy, math
import pandas as pd

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
    plt.title(figurename[17:])
    plt.xlabel("generation")
    plt.ylabel("function value")
    plt.savefig(figurename + ".png")

def get_row_data(data_path):
    files = os.listdir(os.getcwd() + data_path)
    for i, file in enumerate(files):
        if file.find(".txt") < 0:
            del files[i]
    total_results = []
    for i, file in enumerate(files):
        print("%d: %s" % (i, file))
        result = []
        with open(os.getcwd() + data_path + file, 'r') as f:
            generation = []
            score = []
            for line in f.readlines():
                if line[0] == '[':
                    info = line.strip().split('|')
                    generation.append(int(info[-2]))
                    score.append(float(info[-1]))
                elif line[0] == '\n':
                    pass
                else:
                    if len(generation) == 0:
                        result.append(None)
                    else:
                        result.append((generation, score))
                    generation = []
                    score = []
        result.pop(0)
        flag = True
        for i, rst in enumerate(result):
            if rst == None:
                while flag:
                    fk = random.choice(result)
                    if fk != None:
                        flag = False
                result[i] = copy.deepcopy(fk)
        total_results.append(result)
        # 填补数据空缺
        for i, method in enumerate(total_results):
            for one_run in method:
                one_run[0].insert(0, 0)
                one_run[1].insert(0, one_run[1][0])
    return total_results

def to_table(total_record, files):
    '''记录统计信息'''
    best_score_record = []
    mean_score_record = []
    std_record = []
    var_record = []
    for i, method in enumerate(total_record):
        score_rcd = []
        total_score = 0
        for one_run in method:
            score_rcd.append(one_run[1][-1])
        # 每种方法的统计信息
        score_rcd = np.array(score_rcd)
        best_score_record.append(score_rcd.min())
        mean_score_record.append(score_rcd.mean())
        std_record.append(score_rcd.std())
        var_record.append(score_rcd.var())
    data = {"best_score": best_score_record,
            "mean_score": mean_score_record,
            "std_score": std_record,
            "var_score": var_record}
    table = pd.DataFrame(data, index=files)
    return table

def vary_of_generation(row_data, files, generation_num, step):
    '''对于每种方法, 统计最多的记录了几代, 将该代数作为横坐标，
    没有中间代的使用小于等于该代数的得分作为该代得分|一种求均值方法'''
    variations = []
    for file, method in zip(files, row_data):
        # 确定generation刻度
        generations = [i * step for i in range(int(generation_num/step))]
        # score_record = np.zeros(int(generation_num/step))
        score_record = []
        # 统计每个generation的100次平均得分
        for g in generations:
            total_score = 0
            for one_run in method:
                genera = one_run[0]
                scores = one_run[1]
                # if g == 0:
                #     total_score += scores[0]
                #     continue
                # else:
                for i in range(len(genera)):
                    if i + 1 == len(genera) or \
                            (genera[i] <= g and genera[i+1] > g):
                        total_score += scores[i]
                        break
            mean_score = total_score / len(method)
            score_record.append(mean_score)
        generations.append(generation_num)
        score_record.append(score_record[-1])
        dct = {"generation": generations,
               "score": score_record}
        table = pd.DataFrame(dct)
        variations.append(table)
    return variations


# def plot_statics(data_table, labels):
#     plt.figure()
#     sigle_width = 0.2
#     xticks = np.arange(len(funcs))
#     for i, label in enumerate(labels):
#         plt.bar(xticks + i * sigle_width, data_table["mean_score"][i:len(data_table):len(labels)],
#                 label=label, width=sigle_width, align="center")
#     plt.xticks(xticks + 0.3, labels=[func.__name__ for func in funcs], rotation=20)
#     plt.ylim(ymin=7000)
#     plt.legend()
#     plt.show()
#
# def plot_substatics(data_table, labels):
#     for combi in combinations(labels, 2):
#         plt.figure()
#         step_width = 0.4
#         sigle_width = 0.2
#         xticks = np.arange(len(funcs))
#         j = 0
#         for i, label in enumerate(labels):
#             if label == combi[0] or label == combi[1]:
#                 x = xticks + j * step_width
#                 y = np.array(data_table["mean_score"][i:len(data_table):len(labels)])
#                 plt.bar(x, y, label=label, width=sigle_width, align="center")
#                 for a, b in zip(x, y):
#                     plt.text(a, b, "%d"%b, horizontalalignment='center', verticalalignment="baseline", fontsize=10)
#                 j += 1
#         plt.xticks(xticks+sigle_width/2, labels=[func.__name__ for func in funcs], fontsize=10, rotation=15)
#         plt.ylim(ymin=7000)
#         plt.legend()
#         plt.show()

def bar_on_mutation(data_table, funcs, labels, single_width, size, rotation=0):
    for i in range(len(funcs)):
        plt.figure(figsize=size)
        xticks = np.arange(len(labels))*single_width
        for j, label in enumerate(labels):
            x = xticks[j]
            y = data_table["mean_score"][len(labels)*i + j]
            plt.bar(x, y, width=single_width, align="center")
            plt.text(x, y, "%.2f" % y, horizontalalignment='center', verticalalignment="baseline", fontsize=10)
        col_labels = labels
        row_labels = ["best score"]
        # plt.table(cellText=[data_table["mean_score"][len(labels)*i:len(labels)*(i+1)]],
        #           rowLabels=row_labels, colLabels=col_labels, colWidths=[0.1]*3,
        #           loc="best")
        plt.xticks(xticks, labels=labels, rotation=rotation)
        plt.title(funcs[i])
        if funcs[i] != "shekel":
            plt.yscale("log")
        plt.savefig("graph\\"+funcs[i])
        plt.show()

def plot_generation_score(generation_score, funcs, labels):
    step = len(labels)
    for i in range(len(funcs)):
        plt.figure()
        for j in range(len(labels)):
            x = generation_score[i*len(labels)+j]["generation"]
            y = generation_score[i*len(labels)+j]["score"]
            plt.plot(x, y, label=labels[j])
            # plt.plot(x, y, drawstyle='steps-post', label=labels[j])
        plt.title(funcs[i])
        plt.legend()
        plt.xscale("log")
        plt.savefig("graph\\" + funcs[i]+"_step")
        plt.show()

def mutator_analysis():
    GENERATION_NUM = 1500
    STEP = 10
    DATA_PATH = "\\data\\selection_and_mutation\\round-robin-mutation\\"

    files = os.listdir(os.getcwd() + DATA_PATH)
    for i, file in enumerate(files):
        if file.find(".txt") < 0:
            del files[i]
    total_results = get_row_data(DATA_PATH)
    data_table = to_table(total_results, files)
    generation_score = vary_of_generation(total_results, files, GENERATION_NUM, STEP)

    labels = ["cauchy", "gaussian", "levy"]
    funcs = ["ackley", "gold_stein_price", "quartic_noise", "rastrigin", "shekel", "step"]

    data_table.to_csv("graph\\mutators.csv")
    bar_on_mutation(data_table, funcs, labels, single_width=0.3, size=(2.5, 5))
    plot_generation_score(generation_score, funcs, labels)


def selector_analysis():
    GENERATION_NUM = 1500
    STEP = 10
    DATA_PATH = "\\data\\selection_and_mutation\\cauchy-selection\\"

    files = os.listdir(os.getcwd() + DATA_PATH)
    for i, file in enumerate(files):
        if file.find(".txt") < 0:
            del files[i]
    total_results = get_row_data(DATA_PATH)
    data_table = to_table(total_results, files)
    generation_score = vary_of_generation(total_results, files, GENERATION_NUM, STEP)

    labels = ["rank", "woulette wheel", "round robin", "trucked"]
    funcs = ["ackley", "gold_stein_price", "quartic_noise", "rastrigin", "shekel", "step"]

    data_table.to_csv("graph\\selector.csv")
    bar_on_mutation(data_table, funcs, labels, single_width=0.3, size=(3, 5))
    plot_generation_score(generation_score, funcs, labels)

def recombinator_analysis():
    GENERATION_NUM = 1500
    STEP = 10
    DATA_PATH = "\\data\\selection_and_mutation\\recombination\\"

    files = os.listdir(os.getcwd() + DATA_PATH)
    for i, file in enumerate(files):
        if file.find(".txt") < 0:
            del files[i]
    total_results = get_row_data(DATA_PATH)
    data_table = to_table(total_results, files)
    generation_score = vary_of_generation(total_results, files, GENERATION_NUM, STEP)

    labels = ["arithmetic", "discrete", "global-discrete"]
    funcs = ["ackley", "gold_stein_price", "quartic_noise", "rastrigin", "shekel", "step"]
    data_table.to_csv("graph\\recombinator.csv")

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


def show_all(dir):
    files = os.listdir(dir)
    generation_size = {"ackley": 101, "gold_stein_price": 7, "quartic_noise": 201,
                       "rastrigin": 334, "shekel": 7, "step": 101}
    show_data = ['023','013','012']
    for file in files:
        if file[0] in show_data[0] and file[1] in show_data[1] and file[2] in show_data[2]:
            plt.figure()
            func = file[4:-13]
            generation, fitness = evolve_process(dir+file, generation_size[func], 10)
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

def performance_table(dir):
    min_table = {"function":[], "ackley":[], "gold_stein_price":[], "quartic_noise":[],
                 "rastrigin":[], "shekel":[], "step":[]}
    mean_table = {"function":[], "ackley":[], "gold_stein_price":[], "quartic_noise":[],
                 "rastrigin":[], "shekel":[], "step":[]}
    std_table = {"function":[], "ackley":[], "gold_stein_price":[], "quartic_noise":[],
                 "rastrigin":[], "shekel":[], "step":[]}
    generation_size = {"ackley":101, "gold_stein_price":7, "quartic_noise":201,
                 "rastrigin":334, "shekel":7, "step":101}
    files = os.listdir(dir)
    pre_function = "__"
    for file in files:
        func = file[4:-13]
        vec = performance_vector(dir+file, generation_size[func], 10)
        if pre_function != file[:3]:
            min_table["function"].append(file[:3])
            mean_table["function"].append(file[:3])
            std_table["function"].append(file[:3])
            pre_function = file[:3]
        min_table[func].append(vec[0])
        mean_table[func].append(vec[1])
        std_table[func].append(vec[2])
    # return min_table, mean_table, std_table
    return pd.DataFrame(min_table), pd.DataFrame(mean_table), \
               pd.DataFrame(std_table)

if __name__ == "__main__":
    print(os.getcwd())
    dir = "..\\data\\newEA\\"
    min_table, mean_table, std_table = performance_table(dir)
    # show_all(dir)








