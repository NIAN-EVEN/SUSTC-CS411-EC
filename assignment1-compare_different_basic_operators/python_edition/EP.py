import time
import sys, os
from assignment1.python_edition.benchmark import *
from basic_definations_and_operators.Mutation import *
from basic_definations_and_operators.Crossover import *
from multiprocessing import Pool

# 023 submit edition
selection_set = [roulette_wheel_select, rank_based_select,
                 round_robin_select, tournament_select, truncated_select]
# 013
mutation_set = [cauchy_mutate, gaussian_mutate, uniform_mutate, one_bit_mutate]
# 012
recombination_set = [arithmetic_crossover, discrete_crossover,
                     global_discrete_crossover, one_bit_crossover]


config = {"pop_size": 100, "mutation_rate": 0.5,
          "parent_size": 4, "offspring_size": 100,
           "crossover_param": 0.5, "selection_param": 10}


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
        pop = selection(pop, pop_size, q)
        pop.sort(key=lambda x: x.fitness)
        generation += 1
        if generation % 10 == 0:
            record["generation"].append(generation)
            record["fitness"].append(pop[0].fitness)
            # print("generation=%d, fitness=%f" % (generation, pop[0].fitness))
            msg = "generation=%d, fitness=%f\n" % (generation, pop[0].fitness)
            # tofile(filename, msg)
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

def task2(pop_list, filename, benchmark, budget, config, selection, mutation):
    print(filename[8:-4])
    best_score = 100
    total_record = {"generation": [], "fitness": []}
    for k, pop in enumerate(pop_list):
        msg = "[%d run, %s, %s, %s]\n" % (k, benchmark[0].__name__,
                             selection.__name__, mutation.__name__)
        # tofile(filename, msg)
        record = EP(benchmark, budget, config, selection, mutation, filename, pop)
        print(record["fitness"][-1])
    #     if record["fitness"][-1] < best_score:
    #         best_score = record["fitness"][-1]

def dbg():
    print(os.getcwd())
    iterNum = 1
    start = time.time()
    for benchmark, budget in zip(benchmark_lst, moreevaluations):
        # for benchmark, budget in zip(benchmark_lst, evaluations):
        # print(benchmark[0].__name__.center(30, "="))
        pop_size = config["pop_size"]
        Solution.dimension = benchmark[1]
        Solution.lower_bound = benchmark[2]
        Solution.upper_bound = benchmark[3]
        pop_list = []
        for i in range(iterNum):
            pop_list.append(init(pop_size, benchmark[0]))
        for s in range(2,3):
            for m in range(2):
                filename = "../data/%d%d_" % (s, m) + benchmark[0].__name__ + ".txt"
                task2(copy.deepcopy(pop_list), filename, benchmark,
                          budget, config, selection_set[s], mutation_set[m])
                # for r in range(3):
                #     filename = "../data/%d%d%d_" % (s, m, r) + benchmark[0].__name__ + ".txt"
                #     task1(copy.deepcopy(pop_list), filename, benchmark,
                #           budget, config, selection_set[s], recombination_set[r],
                #           mutation_set[m])
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
                filename = "../data/%d%d_" % (s, m) + benchmark[0].__name__ + ".txt"
                p.apply_async(task2, args=(copy.deepcopy(pop_list), filename, benchmark,
                                           budget, config, selection_set[s], mutation_set[m]))
    p.close()
    p.join()
    print("running %f sconds" % (time.time() - start))

if __name__ == "__main__":
    argv1 = sys.argv[1]
    if argv1 == "dbg":
        dbg()
    else:
        on_server()

