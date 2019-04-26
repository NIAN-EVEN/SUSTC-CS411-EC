import numpy as np
import sys, random, os, time, copy
from math import isnan
from ..assignment1.python_edition.benchmark import *

def optimise(pop, benchmark, budget, config, selection, recombination, mutation, filename):
    func = benchmark[0]
    Solution.dimension = benchmark[1]
    Solution.lower_bound = benchmark[2]
    Solution.upper_bound = benchmark[3]
    Cr = config["Cr"]
    F = config["F"]
    evaluate_num = 0
    pop_size = config["pop_size"]
    evaluate_num += pop_size
    generation = 0
    record = {"generation": [0],
              "fitness": [pop[0].fitness]}
    while evaluate_num < budget:
        for i, p in enumerate(pop):
            rand_num = np.random.randint(0, len(pop), size=5).tolist()
            while i in rand_num:
                idxi = rand_num.index(i)
                rand_num[idxi] = np.random.randint(0, len(pop))
            pa, pb, pc, pd, pe = [pop[j] for j in rand_num]
            vec_p = pa.vec + F*(pb.vec - pc.vec) + F*(pd.vec - pe.vec)
            vec_pp = copy.deepcopy(p.vec)
            R = np.random.randint(Solution.dimension)
            for j in range(Solution.dimension):
                if np.random.rand() < Cr or j == R:
                    vec_pp[j] = vec_p[j]
            p_pp = Solution(vec_pp, None)
            p_pp.fitness = fitness(func, vec_pp)
            pop[i] = p_pp if p_pp.fitness < p.fitness else p
            evaluate_num += 1
        pop.sort(key=lambda x: x.fitness)
        generation += 1
        if generation % 10 == 0:
            record["generation"].append(generation)
            record["fitness"].append(pop[0].fitness)
    record["generation"].append(record["generation"][-1]+10)
    record["fitness"].append(pop[0].fitness)
    return record, pop[0]

# ======================= definition =====================================
class Solution():
    dimension = None
    lower_bound = None
    upper_bound = None
    def __init__(self, x, yita, fitness=None):
        self.vec = x
        self.yita = yita
        self.fitness = None

    def is_legal(self):
        for val in self.vec:
            # fixme: 全局变量多进程共享
            # print("in is legal?????")
            # print("this is upper bound %f" % Solution.upper_bound)
            if isnan(val) or val > Solution.upper_bound \
                or val < Solution.lower_bound:
                return False
        # print("yes legal")
        return True

    def correct(self):
        for i, val in enumerate(self.vec):
            if isnan(val) or val > Solution.upper_bound \
                or val < Solution.lower_bound:
                self.vec[i] = np.random.uniform(Solution.lower_bound,
                                                Solution.upper_bound)

    def __len__(self):
        return self.vec.size

    def __str__(self):
        return "x=%s, fitness=%d" % (str(self.vec), self.fitness)

def fitness(func, x):
    objValue = func(x)
    return objValue

def tofile(filename, msg):
    with open(filename, "a") as f:
        f.write(msg)

def init(num, func):
    pop = []
    for i in range(num):
        pop.append(new_solution())
    for p in pop:
        p.fitness = fitness(func, p.vec)
    pop.sort(key=lambda x: x.fitness)
    return pop

def new_solution():
    x = np.random.uniform(Solution.lower_bound,
                          Solution.upper_bound,
                          size=Solution.dimension)
    yita = np.zeros_like(x) + 3.0
    return Solution(x, yita)

#======================== differential  ===================================


#======================== selection ===================================
def roulette_wheel_selection(pop, n, _):
    max_fitness = max(pop, key=lambda p:p.fitness).fitness
    roulette_wheel = []
    for p in pop:
        # scaled_fitness = max_fitness-p.fitness
        scaled_fitness = np.log(max_fitness-p.fitness+1)
        roulette_wheel.append(scaled_fitness)
    sum_fitness = sum(roulette_wheel)
    selected_groups = []
    for i in range(n):
        rand_num = np.random.rand() * sum_fitness
        record = 0
        for j, scaled_fitness in enumerate(roulette_wheel):
            if record <= rand_num and rand_num <= record + scaled_fitness:
                selected_groups.append(pop[j])
                break
            record += scaled_fitness
    return selected_groups

def round_robin_selection(pop, n, q=10):
    '''
    selection method in CEP and FEP
    :param pop:
    :param q: select q opponents for compitation
    :param n: select n individuals
    :return: truncated population
    '''
    for p in pop:
        p.win = 0
    for p in pop:
        opponents = random.sample(pop, q)
        for oppo in opponents:
            if p.fitness <= oppo.fitness:
                p.win += 1
            else:
                oppo.win += 1
    pop.sort(key=lambda x: x.win, reverse=True)
    return pop[:n]

def rank_based_selection(pop, n, _):
    '''在pop中选择n个个体'''
    pop.sort(key=lambda x: x.fitness)
    roulette_wheel = []
    sum_fitness = 0
    for i, p in enumerate(pop):
        # scaled_fitness = max_fitness-p.fitness
        scaled_fitness = i**2
        roulette_wheel.append(scaled_fitness)
        sum_fitness += scaled_fitness
    selected_groups = []
    for i in range(n):
        rand_num = np.random.rand() * sum_fitness
        record = 0
        for j, scaled_fitness in enumerate(roulette_wheel):
            if record <= rand_num and record + scaled_fitness > rand_num:
                selected_groups.append(pop[j])
                break
            record += scaled_fitness
    return selected_groups

def tournament_selection(pop, n, q=10):
    selected_group = []
    for i in range(n):
        opponents = random.sample(pop, q)
        opponents.sort(key=lambda x: x.fitness)
        selected_group.append(opponents[0])
    return selected_group

# ===================== mutation ==================================
def cauchy_mutation(solution):
    # print("in cauchy")
    x = solution.vec
    yita = solution.yita
    dim = x.size
    tao = 1 / np.sqrt(2 * np.sqrt(dim))
    tao_prime = 1 / np.sqrt(2 * dim)
    # print("real in cauchy")
    x_prime = x + yita*np.random.standard_cauchy(dim)
    yita_prime = yita*np.exp(tao_prime*np.random.standard_normal() \
                             +tao*np.random.standard_normal(dim))
    n_solution = Solution(x_prime, yita_prime)
    # print("end cauchy")
    legal_offspring = n_solution.is_legal()
    # print("ohh cauchy")

    if not legal_offspring:
        n_solution.correct()

    return n_solution

def gaussian_mutation(solution):
    # print("in gaussian")
    x = solution.vec
    yita = solution.yita
    dim = x.size
    tao = 1 / np.sqrt(2 * np.sqrt(dim))
    tao_prime = 1 / np.sqrt(2 * dim)

    x_prime = x+yita*np.random.standard_normal(dim)
    yita_prime = yita*np.exp(tao_prime*np.random.standard_normal() \
                             + tao*np.random.standard_normal(dim))
    new_solution = Solution(x_prime, yita_prime)
    legal_offspring = new_solution.is_legal()

    if not legal_offspring:
        new_solution.correct()
    # print("in gaussian")
    return new_solution

def one_bit_mutation(solution):
    # print("in one bit mutation")
    bit = np.random.randint(0, len(solution)-1)
    new_soluion = copy.deepcopy(solution)
    new_soluion.vec[bit] = np.random.uniform(Solution.lower_bound,
                                             Solution.upper_bound)
    # print('after one bit')
    return new_soluion

# ============================= crossover ==============================
def arithmetic_recombination(solution, _):
    alpha = np.random.rand()
    x1 = solution[0].vec
    yita1 = solution[0].yita
    x2 = solution[1].vec
    yita2 = solution[1].yita
    x1_prime = alpha * x1 + (1-alpha) * x2
    x2_prime = alpha * x2 + (1-alpha) * x1
    yita1_prime = alpha * yita1 + (1 - alpha) * yita2
    yita2_prime = alpha * yita2 + (1 - alpha) * yita1
    return [Solution(x1_prime, yita1_prime), Solution(x2_prime, yita2_prime)]

def discrete_recombination(solution, _):
    # k point crossover
    k = np.random.rand()
    x1 = solution[0].vec
    yita1 = solution[0].yita
    x2 = solution[1].vec
    yita2 = solution[1].yita
    x1_prime = copy.deepcopy(x1)
    yita1_prime = copy.deepcopy(yita1)
    x2_prime = copy.deepcopy(x2)
    yita2_prime = copy.deepcopy(yita2)
    crossover_list = np.random.randint(len(x1), size=int(k*len(x1)))
    for point in crossover_list:
        x1_prime[point] = x2[point]
        yita1_prime[point] = yita2[point]
        x2_prime[point] = x1[point]
        yita2_prime[point] = yita1[point]

    return [Solution(x1_prime, yita1_prime),
            Solution(x2_prime, yita2_prime)]

def global_discrete_recombination(solution, _):
    x_new = np.zeros_like(solution[0].vec)
    yita_new = np.zeros_like(solution[0].vec)
    for i in range(Solution.dimension):
        parent = rank_based_selection(solution, 1, None)
        x_new[i] = parent[0].vec[i]
        yita_new[i] = parent[0].yita[i]
    return [Solution(x_new, yita_new)]

def dbg():
    print(os.getcwd())
    iterNum = 1
    for benchmark, budget in zip(benchmark_lst, evaluations):
        # for benchmark, budget in zip(benchmark_lst, evaluations):
        # print(benchmark[0].__name__.center(30, "="))
        pop_size = config["pop_size"]
        Solution.dimension = benchmark[1]
        Solution.lower_bound = benchmark[2]
        Solution.upper_bound = benchmark[3]
        pop_list = []
        for i in range(iterNum):
            pop_list.append(init(pop_size, benchmark[0]))
        for r in range(len(recombination_set)):
            for m in range(len(mutation_set)):
                for s in range(len(selection_set)):
                    print('[Optimisation of function %s, crossoverIdx=%d, mutationId=%d, selectionIdx=%d]'
                          % (benchmark[0].__name__, r, m, s))
                    appry = np.zeros(iterNum)
                    for k, pop in enumerate(pop_list):
                        record, pop0 = optimise(pop, benchmark, budget, config, selection_set[s],
                                                recombination_set[r], mutation_set[m], filename=None)
                        print('RUN %d: Approximate optimal value=%.16f' % (k, pop0.fitness))
                        print('RUN %d: Approximate optimum=%s' % (k, str(pop0.vec)))
                        appry[k] = pop0.fitness
                    print('FINAL: Averaged approximate optimal value=%.16f(%.16f)\n' %
                          (appry.mean(), appry.std() / np.sqrt(iterNum)))

selection_set = [roulette_wheel_selection, round_robin_selection, tournament_selection]

mutation_set = [cauchy_mutation, gaussian_mutation, one_bit_mutation]

recombination_set = [arithmetic_recombination, discrete_recombination, global_discrete_recombination]

config = {"pop_size": 100, "Cr": 0.5, "F": 0.5}

if __name__ == "__main__":
    dbg()
