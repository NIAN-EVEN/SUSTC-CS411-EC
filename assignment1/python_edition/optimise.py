import numpy as np
from itertools import combinations
import sys, random, os
from benchmark import *
from alg_analysis import *

class Solution():
    def __init__(self, x, yita, fitness):
        self.vec = x
        self.fitness = fitness
        self.yita = yita

    def __len__(self):
        return self.vec.size

def optimise(benchmark, budget, recombination=0, mutation=0, selection=0):
    # get configuration
    func = benchmark[0]
    dimension = benchmark[1]
    lower_bound = benchmark[2]
    upper_bound = benchmark[3]
    mu = 100
    q = 10
    pop = init(mu, dimension, upper_bound, lower_bound, func)
    record_score = [-1 * pop[0].fitness]
    # init population
    # TODO: integrate and debug
    for i in range(budget):
        for p in pop[:mu]:
            pop.append(IFEP_mutation(p, func))
        pairwise_selection(pop, q, mu)
        if -1 * pop[0].fitness < record_score[-1]:
            record_score.append(-1 * pop[0].fitness)
        else:
            record_score.append(record_score[-1])
    save_data(func.__name__, record_score)
    figure_plot(func.__name__, record_score)


def fitness(func, x):
    # 原函数求最小值，fitness 求最大值
    objValue = func(x)
    return -1 * objValue

def init(num, dim, ub, lb, func):
    pop = []
    for i in range(num):
        x = np.random.rand(dim)*(ub - lb) + ub
        yita = np.zeros(dim) + 3.0
        pop.append(Solution(x, yita, fitness(func, x)))
    pop.sort(key=lambda x:x.fitness, reverse=True)
    return pop

def quadratic_recombination(solution, func):
    x1 = solution[0].vec
    fx1= solution[0].fitness
    x2 = solution[1].vec
    fx2= solution[1].fitness
    x3 = solution[2].vec
    fx3= solution[2].fitness
    fenzi = (x2**2-x3**2)*fx1+(x3**2-x1**2)*fx2+(x1**2-x2**2)*fx3
    fenmu =   (x2 - x3) * fx1 + (x3 - x1) * fx2 + (x1 - x2) * fx3
    x4 = (1/2) * (fenzi/fenmu)
    return Solution(x4, None, fitness(func, x4))

def IFEP_mutation(solution, func):
    x1, yita1 = FEP_mutation(solution.vec, solution.yita)
    fitness1 = fitness(func, x1)
    x2, yita2 = CEP_mutation(solution.vec, solution.yita)
    fitness2 = fitness(func, x2)
    if fitness1 >= fitness2:
        return Solution(x1, yita1, fitness1)
    else:
        return Solution(x2, yita2, fitness2)

def FEP_mutation(x, yita):
    dim = x.size
    cauchy_j = np.random.standard_cauchy(dim)
    gaussian_j = np.random.standard_normal(dim)
    gaussian = np.random.standard_normal()
    tao = 1/np.sqrt(2*np.sqrt(dim))
    tao_prime = 1/np.sqrt(2*np.sqrt(dim))

    x_prime = x+yita*cauchy_j
    yita_prime = yita*np.exp(tao_prime*gaussian+tao*gaussian_j)

    return x_prime, yita_prime

def CEP_mutation(x, yita):
    dim = x.size
    gaussian_j = np.random.standard_normal(dim)
    gaussian = np.random.standard_normal()
    tao = 1/np.sqrt(2*np.sqrt(dim))
    tao_prime = 1/np.sqrt(2*np.sqrt(dim))

    x_prime = x+yita*gaussian_j
    yita_prime = yita*np.exp(tao_prime*gaussian+tao*gaussian_j)

    return x_prime, yita_prime

def diverse_selection(pop, group_num, group_size=2, alpha=0.1):
    popSize = len(pop)
    # 计算pop中元素的两两距离
    groups = []
    for pair in combinations(pop, group_size):
        # 计算中心点，group的差异度在于各个点于中心点的距离和
        center = center_of_points([p.vec for p in pop])
        score = 0
        for s in pair:
            score += dist(s.vec, center.vec)
            score += s.fitness
        groups.append((pair, score))
    groups.sort(lambda x:x[1], reverse=True)
    return rank_based_selection(groups, group_num)

def rank_based_selection(pop, n):
    # TODO: search better rank based selection
    pass

def pairwise_selection(pop, q, mu):
    for p in pop:
        p.win = 0
    for p in pop:
        opponents = random.sample(pop, q)
        for oppo in opponents:
            if p.fitness >= oppo.fitness:
                p.win += 1
            else:
                oppo.win += 1
    pop.sort(key=lambda x:x.win, reverse=True)
    del pop[mu:]

def center_of_points(X):
    point_num = len(X)
    sum = np.zeros(len(X[0]))
    for x in X:
        sum += x
    return sum/point_num

def dist(x, y):
    dim = len(x)
    distance = 0
    for d in dim:
        distance += (x[d]-y[d])**2
    return np.sqrt(distance)

if __name__ == "__main__":
    for benchmark, budget in zip(benchmark_lst, generation_lst):
        optimise(benchmark, budget)