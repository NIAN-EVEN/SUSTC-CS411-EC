import sys
sys.path.append("../")

from basic_definations_and_operators.Solution import *
from basic_definations_and_operators.Crossover import *
from basic_definations_and_operators.Mutation import *
from assignment2_compare_different_niching_methods.cec2013.cec2013 import *
import random

def crowding_update(pop, crossover, mutate, param, func_id=None, **kw):
    best_so_far = pop[0]
    opt_cnt = []
    generation = 0
    evaluation_num = param["pop_size"]
    while evaluation_num < param["evaluation_ub"]:
        father_idx, mather_idx = nearest_pairing(pop)
        for fidx, midx in zip(father_idx, mather_idx):
            p1 = pop[fidx]
            p2 = pop[midx]
            c1, c2 = crossover([p1, p2], param["crossover_alpha"])
            evaluation_num += 2
            if c1.fitness <= p1.fitness and c1.fitness <= p2.fitness:
                c1 = mutate(c1)
                evaluation_num += 1
            if c2.fitness <= p1.fitness and c2.fitness <= p2.fitness:
                c2 = mutate(c2)
                evaluation_num += 1
            if distance(p1, c1)+distance(p2, c2) <= \
                distance(p1, c2) + distance(p2, c1):
                if c1.fitness > p1.fitness:
                    c1.generation = generation
                    pop[fidx] = c1
                if c2.fitness > p2.fitness:
                    c2.generation = generation
                    pop[midx] = c2
            else:
                if c2.fitness > p1.fitness:
                    c2.generation = generation
                    pop[fidx] = c2
                if c1.fitness > p2.fitness:
                    c1.generation = generation
                    pop[midx] = c1
        pop.sort(key=lambda x: x.fitness, reverse=True)
        ################################## record #####################################
        if generation % 10 == 0:
            X = np.array([p.x for p in pop])
            count, seeds = how_many_goptima(X, CEC2013(func_id), 0.0001)
            opt_cnt.append(count)
        ################################## record #####################################
        generation += 1
        # ################################# test ########################################
        # pop.sort(key=lambda x: x.real_fitness, reverse=True)
        # if compare(pop[0].real_fitness, best_so_far.real_fitness) or generation % 50 == 0:
        #     if compare(pop[0].real_fitness, best_so_far.real_fitness):
        #         best_so_far = pop[0]
        #     print("evaluation=%d, %f" % (evaluation_num, best_so_far.real_fitness))
        # ################################# test ########################################
    X = np.array([p.x for p in pop])
    count, seeds = how_many_goptima(X, CEC2013(func_id), 0.0001)
    opt_cnt.append(count)
    return opt_cnt

def crowding_update_mutate(pop, crossover, mutate, param, func_id, **kw):
    best_so_far = pop[0]
    opt_cnt = []
    generation = 0
    evaluation_num = param["pop_size"]
    while evaluation_num < param["evaluation_ub"]:
        father_idx, mather_idx = nearest_pairing(pop)
        for fidx, midx in zip(father_idx, mather_idx):
            p1 = pop[fidx]
            p2 = pop[midx]
            c1, c2 = crossover([p1, p2], param["crossover_alpha"])
            if np.random.rand() < param["mutation_rate"]:
                if np.random.rand() < 0.5:
                    c1 = mutate(c1)
                else:
                    c2 = mutate(c2)
            evaluation_num += 2
            if distance(p1, c1)+distance(p2, c2) <= \
                distance(p1, c2) + distance(p2, c1):
                if c1.fitness > p1.fitness:
                    c1.generation = generation
                    pop[fidx] = c1
                if c2.fitness > p2.fitness:
                    c2.generation = generation
                    pop[midx] = c2
            else:
                if c2.fitness > p1.fitness:
                    c2.generation = generation
                    pop[fidx] = c2
                if c1.fitness > p2.fitness:
                    c1.generation = generation
                    pop[midx] = c1
        pop.sort(key=lambda x: x.fitness, reverse=True)
        # ################################## record #####################################
        # if generation % 10 == 0:
        #     X = np.array([p.x for p in pop])
        #     count, seeds = how_many_goptima(X, CEC2013(func_id), 0.0001)
        #     opt_cnt.append(count)
        # ################################## record #####################################
        generation += 1
        ################################# test ########################################
        # if compare(pop[0].real_fitness, best_so_far.real_fitness) or generation % 50 == 0:
        #     if compare(pop[0].real_fitness, best_so_far.real_fitness):
        #         best_so_far = pop[0]
        #     print("evaluation=%d, %f" % (evaluation_num, best_so_far.real_fitness))
        if generation % 10 == 0:
            X = np.array([p.x for p in pop])
            count, seeds = how_many_goptima(X, CEC2013(func_id), 0.01)
            print("accuracy: %f, count: %d" % (0.01, count))
        ################################# test ########################################
    X = np.array([p.x for p in pop])
    count, seeds = how_many_goptima(X, CEC2013(func_id), 0.01)
    opt_cnt.append(count)
    return opt_cnt

if __name__ == "__main__":
    # function_idx = [1, 6, 11, 13]
    function_idx = [11, 13]
    accuracy_level = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    result = []
    for func_idx in function_idx:
        func = CEC2013(func_idx)
        param = {"pop_size": 100,
                 "evaluation_ub": func.get_maxfes(),
                 "crossover_alpha": 0.6,
                 "mutation_rate": 0.5}
        Solution.dimension = func.get_dimension()
        Solution.lower_bound = func.get_lbound(0)
        Solution.upper_bound = func.get_ubound(0)
        Solution.fitness_function = func.evaluate
        pop = init(param["pop_size"])
        crowding_update(pop=pop,
                        crossover=heuristic_crossover,
                        mutate=one_bit_mutate,
                        param=param,
                        func_id=func_idx)
        X = np.array([p.x for p in pop])
        for accuracy in accuracy_level:
            count, seeds = how_many_goptima(X, func, accuracy)
            print("In the current population there exist", count, "global optimizers.")
            # print("Global optimizers: \n", seeds)
            result.append(count)
    for func_idx, opt_cnt, accuracy in zip(function_idx, result, accuracy_level):
        print("function:%d, opt_cnt:%d, real_opt:%d, accuracy:%f" %
              (func_idx, opt_cnt, CEC2013(func_idx).get_no_goptima(), accuracy))