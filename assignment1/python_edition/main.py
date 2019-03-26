import numpy as np
import time
import pandas as pd
from itertools import combinations
import sys, random, os
from benchmark import *
from defination import *
from selection import *
from mutation import *
from recombination import *
from multiprocessing import Pool

# 023 submit edition
# selection_set = [roulette_wheel_selection, rank_based_selection,
#                  round_robin_selection, tournament_selection, truncated_selection]
# test
selection_set = [roulette_wheel_selection,round_robin_selection, tournament_selection]
# 013
mutation_set = [cauchy_mutation, gaussian_mutation, uniform_mutation, one_bit_mutation]
# 012
recombination_set = [arithmetic_recombination, discrete_recombination,
                     global_discrete_recombination, one_bit_crossover]


config = {"pop_size": 100, "mutation_rate": 0.5,
          "parent_size": 4, "offspring_size": 100,
           "crossover_param": 0.5, "selection_param": 10}

def optimise(pop, benchmark, budget, config, selection, recombination, mutation, filename):
    # get configuration
    func = benchmark[0]
    Solution.dimension = benchmark[1]
    Solution.lower_bound = benchmark[2]
    Solution.upper_bound = benchmark[3]
    mutation_rate = config["mutation_rate"]
    offspring_size = config["offspring_size"]
    parent_size = config["parent_size"]
    q = config["selection_param"]
    crossover_param = config["crossover_param"]
    evaluate_num = 0
    pop_size = config["pop_size"]
    evaluate_num += pop_size
    record = {"generation": [0],
              "fitness": [pop[0].fitness]}
    # init population
    generation = 0
    msg = "generation=%d, fitness=%f\n" % (generation, pop[0].fitness)
    tofile(filename, msg)
    # print("OPT1")
    while evaluate_num < budget:
    # while generation <= 1500:
    #     if generation % 100 == 0:
    #         print("generation=%d, fitness=%f" % (generation, pop[0].fitness))
        offsprings = []
        # 通过crossover产生定长的offspring
        # print("OPT2")
        while len(offsprings) < offspring_size:
            parents = selection(pop, parent_size, q)
            for p in recombination(parents, crossover_param):
                p.fitness = fitness(func, p.vec)
                offsprings.append(p)
        # submit edition
        # mutation_offsprings = selection(offsprings, int(offspring_size*mutation_rate), q)
        # testing edition
        offsprings.sort(key=lambda x: x.fitness, reverse=True)
        mutation_offsprings = offsprings[:int(offspring_size*mutation_rate)]
        for off in mutation_offsprings:
            p = mutation(off)
            # print("%d after mutation+++++++++++++++++++" % evaluate_num)
            p.fitness = fitness(func, p.vec)
            offsprings.append(p)
        # print("OPT3")
        evaluate_num += len(offsprings)
        pop.extend(offsprings)
        pop.sort(key=lambda x: x.fitness)
        pop =  pop[:pop_size]
        generation += 1
        if generation % 10 == 0:
            record["generation"].append(generation)
            record["fitness"].append(pop[0].fitness)
            # print("generation=%d, fitness=%f" % (generation, pop[0].fitness))
            msg = "generation=%d, fitness=%f\n" % (generation, pop[0].fitness)
            tofile(filename, msg)
        # print("OPT4")
    record["generation"].append(record["generation"][-1]+10)
    record["fitness"].append(pop[0].fitness)
    # print("OPT5")
    return record

def EP(benchmark, budget, config, selection, mutation, filename, pop):
    # get configuration
    func = benchmark[0]
    Solution.dimension = benchmark[1]
    Solution.lower_bound = benchmark[2]
    Solution.upper_bound = benchmark[3]
    q = config["selection_param"]
    evaluate_num = 0
    pop_size = config["pop_size"]
    evaluate_num += pop_size
    record = {"generation": [0],
              "fitness": [pop[0].fitness]}
    # init population
    generation = 0
    # print("EP1")
    while evaluate_num < budget:
    # while generation <= 1500:
    #     if generation % 100 == 0:
    #         print("generation=%d, fitness=%f" % (generation, pop[0].fitness))
        offsprings = []
        # 通过crossover产生定长的offspring
        for p in pop:
            off = mutation(p)
            off.fitness = fitness(func, off.vec)
            offsprings.append(off)
        # print("in EP2")
        evaluate_num += len(offsprings)
        pop.extend(offsprings)
        pop.sort(key=lambda x: x.fitness)
        pop =  pop[:pop_size]
        generation += 1
        if generation % 10 == 0:
            record["generation"].append(generation)
            record["fitness"].append(pop[0].fitness)
            # print("generation=%d, fitness=%f" % (generation, pop[0].fitness))
            msg = "generation=%d, fitness=%f\n" % (generation, pop[0].fitness)
            tofile(filename, msg)
        # print("in EP3")
    record["generation"].append(record["generation"][-1]+10)
    record["fitness"].append(pop[0].fitness)
    # print("in EP4")
    return record

def record_handler(total_record, record):
    # todo: there is some problems
    if len(total_record["generation"]) == 0:
        total_record["generation"].extend(record["generation"])
        total_record["fitness"].extend(record["fitness"])
    elif len(total_record["generation"]) < len(record["generation"]):
        while len(total_record["generation"]) < len(record["generation"]):
            total_record["generation"].append(total_record["generation"][-1] + 10)
            total_record["fitness"].append(total_record["fitness"][-1])
        for i in range(len(total_record["generation"])):
            total_record["fitness"][i] += record["fitness"][i]
    elif len(total_record["generation"]) > len(record["generation"]):
        while len(total_record["generation"]) > len(record["generation"]):
            record["generation"].append(record["generation"][-1] + 10)
            record["fitness"].append(record["fitness"][-1])
        for i in range(len(total_record["generation"])):
            total_record["fitness"][i] += record["fitness"][i]
    # print(total_record)
    # return total_record

def task1(pop_list, filename, benchmark, budget, config, selection, recombination, mutation):
    print("on processing %s" % filename[8:-4])
    best_score = sys.float_info.max
    total_record = {"generation": [], "fitness": []}
    for k, pop in enumerate(pop_list):
        msg = "[%d run, %s, %s, %s, %s]\n" % (k, benchmark[0].__name__, selection.__name__,
                                            recombination.__name__, mutation.__name__)
        tofile(filename, msg)
        record = optimise(pop, benchmark, budget, config, selection, recombination, mutation, filename)
    #     if record["fitness"][-1] < best_score:
    #         best_score = record["fitness"][-1]
    #     record_handler(total_record, record)
    # msg = "[best score in %d run is %f]\n" % (len(pop_list), best_score)
    # tofile(filename, msg)
    # mean_record = {"generation": [], "fitness": []}
    # pop_num = len(pop_list)
    # for i in range(len(total_record["generation"])):
    #     mean_record["generation"].append(total_record["generation"][i])
    #     mean_record["fitness"].append(total_record["fitness"][i]/pop_num)
    # msg = "[mean score in %d run is %f]\n" % (len(pop_list), mean_record["fitness"][-1])
    # tofile(filename, msg)
    # df = pd.DataFrame(mean_record)
    # df.to_csv("../data/%s.csv" % (filename[:-4]))

def task2(pop_list, filename, benchmark, budget, config, selection, mutation):
    print("on processing %s" % filename[8:-4])
    best_score = 100
    total_record = {"generation": [], "fitness": []}
    for k, pop in enumerate(pop_list):
        msg = "[%d run, %s, %s, %s]\n" % (k, benchmark[0].__name__,
                             selection.__name__, mutation.__name__)
        tofile(filename, msg)
        record = EP(benchmark, budget, config, selection, mutation, filename, pop)
        if record["fitness"][-1] < best_score:
            best_score = record["fitness"][-1]
        record_handler(total_record, record)
    msg = "[best score in %d run is %f]\n" % (len(pop_list), best_score)
    tofile(filename, msg)
    mean_record = {"generation": [], "fitness": []}
    pop_num = len(pop_list)
    for i in range(len(total_record["generation"])):
        mean_record["generation"].append(total_record["generation"][i])
        mean_record["fitness"].append(total_record["fitness"][i] / pop_num)
    msg = "[mean score in %d run is %f]\n" % (len(pop_list), mean_record["fitness"][-1])
    tofile(filename, msg)
    df = pd.DataFrame(mean_record)
    df.to_csv("../data/%s.csv" % (filename[:-4]))

def dbg():
    print(os.getcwd())
    iterNum = 10
    start = time.time()
    for benchmark, budget in zip(benchmark_lst[4:5], moreevaluations[4:5]):
        # for benchmark, budget in zip(benchmark_lst, evaluations):
        # print(benchmark[0].__name__.center(30, "="))
        pop_size = config["pop_size"]
        Solution.dimension = benchmark[1]
        Solution.lower_bound = benchmark[2]
        Solution.upper_bound = benchmark[3]
        pop_list = []
        for i in range(iterNum):
            pop_list.append(init(pop_size, benchmark[0]))
        for s in range(len(selection_set)):
            for m in range(2):
                # filename = "../data/%d%d_" % (s, m) + benchmark[0].__name__ + ".txt"
                # task2(copy.deepcopy(pop_list), filename, benchmark,
                #                            budget, config, selection_set[s], mutation_set[m]))
                for r in range(3):
                    filename = "../data/%d%d%d_" % (s, m, r) + benchmark[0].__name__ + ".txt"
                    task1(pop_list, filename, benchmark, budget, config, selection_set[s], recombination_set[r],
                          mutation_set[m])
    print("running %f sconds" % (time.time() - start))

def on_server():
    p = Pool()
    iterNum = 10
    start = time.time()
    for benchmark, budget in zip(benchmark_lst, moreevaluations):
        pop_size = config["pop_size"]
        Solution.dimension = benchmark[1]
        Solution.lower_bound = benchmark[2]
        Solution.upper_bound = benchmark[3]
        pop_list = []
        for i in range(iterNum):
            pop_list.append(init(pop_size, benchmark[0]))
        for s in range(len(selection_set)):
            for m in range(2):
                # filename = "../data/%d%d_" % (s, m) + benchmark[0].__name__ + ".txt"
                # p.apply_async(task2, args=(copy.deepcopy(pop_list), filename, benchmark,
                #                            budget, config, selection_set[s], mutation_set[m])
                for r in range(3):
                    filename = "../data/%d%d%d_" % (s, m, r) + benchmark[0].__name__ + ".txt"
                    p.apply_async(task1, args=(copy.deepcopy(pop_list), filename,
                                               benchmark, budget, config, selection_set[s],
                                               recombination_set[r], mutation_set[m]))
    p.close()
    p.join()
    print("running %f sconds" % (time.time() - start))

if __name__ == "__main__":
    argv1 = sys.argv[1]
    if argv1 == "dbg":
        dbg()
    else:
        on_server()

