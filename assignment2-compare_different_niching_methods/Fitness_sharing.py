import sys
sys.path.append("../")
from basic_definations_and_operators.Solution import *
from basic_definations_and_operators.Crossover import *
from basic_definations_and_operators.Mutation import *
from basic_definations_and_operators.Selection import *
from assignment2.cec2013.cec2013 import *
import random

def EA_niching(pop, crossover, mutate, param, func_id):
    opt_cnt = []
    evaluation_num = param["pop_size"]
    best_so_far = pop[0]
    generation = 1
    while evaluation_num < param["evaluation_ub"]:
        pop = explicit_niching_fitness_rescale(pop, param["sigma_share"],
                                               param["niching_alpha"], param["niching_beta"])
        father_idx, mather_idx = nearest_pairing(pop)
        offspring = []
        for fidx, midx in zip(father_idx, mather_idx):
            p1 = pop[fidx]
            p2 = pop[midx]
            c1, c2 = crossover([p1, p2], param["crossover_alpha"])
            evaluation_num += 2
            if c1.real_fitness <= p1.real_fitness and c1.real_fitness <= p2.real_fitness:
                c1 = mutate(c1)
                evaluation_num += 1
            if c2.real_fitness <= p1.real_fitness and c2.real_fitness <= p2.real_fitness:
                c2 = mutate(c2)
                evaluation_num += 1
            offspring.append(c1)
            offspring.append(c2)
        for p in offspring:
            p.generation = generation
        pop.extend(offspring)
        pop = explicit_niching_fitness_rescale(pop, param["sigma_share"],
                                               param["niching_alpha"], param["niching_beta"])
        pop.sort(key=lambda x: x.fitness, reverse=True)
        pop = pop[:param["pop_size"]]
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


# 是否直接在fitness上做文章没有合理性？既然轮盘赌不是很好，sharing不是很好
def explicit_niching_fitness_rescale(pop, sigma_share, alpha=1, beta=2):
    '''
    niching 方法计算种群fitness_share
    :param pop:
    :param sigma_share: share radius
    :param alpha: scale parameter
    :return: pop with fitness_share
    '''
    # 计算distance table
    distance_table = distance_matrix(pop)
    sh_table = sharing_matrix(distance_table, sigma_share, alpha)
    # 计算sharing
    for i, p in enumerate(pop):
        p.sharing = sum(sh_table[i])
        p.fitness = pow(p.real_fitness, beta) / p.sharing
    return pop

def fitness_rescale(pop, p, sigma_share, alpha=1, beta=2):
    distance_vector = np.array([distance(p, q) for q in pop])
    sh_vector = np.array([sh_function(distance_vector[i],
                                      sigma_share, alpha) for i in range(len(pop))])
    p.sharing = sh_vector.sum()
    p.fitness = pow(p.real_fitness, beta) / p.sharing

def paring_select(pop, select):
    father = select(pop, 1)[0]
    if pop[0] is not father:
        min_distance = distance(father, pop[0])
        mather = pop[0]
    else:
        min_distance = distance(father, pop[1])
        mather = pop[1]
    for p in pop:
        if distance(father, p) < min_distance and p is not father:
            mather = p
    return [father, mather]

def sharing_matrix(distance_matrix, sigma_share, alpha):
    pop_size = distance_matrix.shape[0]
    sh_table = np.zeros((pop_size, pop_size))
    for i in range(pop_size):
        for j in range(i, pop_size):
            sh_table[i, j] = sh_function(distance_matrix[i, j],
                                            sigma_share, alpha)
            sh_table[j, i] = sh_table[i, j]
    return sh_table

def sh_function(d, sigma_share, alpha):
    if d < sigma_share:
        return 1 - pow(d/sigma_share, alpha)
    else:
        return 0

if __name__ == "__main__":
    # function_idx = [1, 6, 9, 11]
    function_idx = [11, 13]
    accuracy_level = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    result = []
    for func_idx in function_idx:
        func = CEC2013(func_idx)
        param = {"pop_size": 500,
                 "evaluation_ub": func.get_maxfes(),
                 "sigma_share": func.get_rho(),
                 "niching_alpha": 3,
                 "niching_beta": 3,
                 "crossover_alpha": 0.6,
                 "speciation_sigma_mate": func.get_rho()
                 }
        Solution.dimension = func.get_dimension()
        Solution.lower_bound = func.get_lbound(0)
        Solution.upper_bound = func.get_ubound(0)
        Solution.fitness_function = func.evaluate
        pop = init(param["pop_size"])
        EA_niching(pop=pop,
                   crossover=heuristic_crossover,
                   mutate=one_bit_mutate,
                   param=param,
                   func_id=func_idx)
        X = np.array([p.x for p in pop])
        for accuracy in accuracy_level:
            count, seeds = how_many_goptima(X, func, accuracy)
            print("In the current population there exist", count, "global optimizers.")
            print("Global optimizers: \n", seeds)
            result.append(count)
    for func_idx, opt_cnt, accuracy in zip(function_idx, result, accuracy_level):
        print("function:%d, opt_cnt:%d, real_opt:%d, accuracy:%f" %
              (func_idx, opt_cnt, CEC2013(func_idx).get_no_goptima(), accuracy))