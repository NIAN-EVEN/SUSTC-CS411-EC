import sys
sys.path.append("../")
from basic_definations_and_operators.Solution import *
'''
selection operator, interface is
1. population
2. select number
3. selection method parameters
return a selected list
'''
from itertools import combinations
import numpy as np
import random

def roulette_wheel_select(pop, n, _):
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

def rank_based_select(pop, n, _):
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
    ###################################################################
    # #simple rank based method
    # pop_size = len(pop)
    # pop.sort(key=lambda x: x.fitness)
    # sum_rank = pop_size * (pop_size + 1) / 2
    # # rand为0-range的随机数
    # rand = np.random.uniform(0, sum_rank, n)
    # # 在整个range中j占比∑(1~j-1)-∑(1~j)部分
    # # 对rand反向求是哪个数累加而成再加1即实现按照排序选择的功能x
    # reverseParentsPos = (np.sqrt(8 * rand + 1) - 1) / 2 + 1
    # selected_groups = []
    # for reverseParent in reverseParentsPos.tolist():
    #     selected_groups.append(pop[len(pop) - int(reverseParent)])
    # return selected_groups
    ###################################################################

def round_robin_select(pop, n, q=10):
    '''
    select method in CEP and FEP
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
            if compare(p.fitness, oppo.fitness):
                p.win += 1
            else:
                oppo.win += 1
    pop.sort(key=lambda x: x.win, reverse=True)
    return pop[:n]

def tournament_select(pop, n, q=10):
    selected_group = []
    for i in range(n):
        opponents = random.sample(pop, q)
        opponents.sort(key=lambda x: x.fitness)
        selected_group.append(opponents[0])
    return selected_group

def truncated_select(pop, n, _):
    pop.sort(key=lambda x: x.fitness)
    return pop[:n]

#######################################################################################
# 复杂度过高
def diverse_select(pop, group_num, group_size=2, alpha=0.1):
    '''
    use sum of distance between points and its center, and sum of group point fitness
    :param pop:
    :param group_num:
    :param group_size:
    :param alpha:
    :return:
    '''
    # TODO: test
    popSize = len(pop)
    # 计算pop中元素的两两距离
    groups = []
    for pair in combinations(pop, group_size):
        # 计算中心点，group的差异度在于各个点于中心点的距离和
        center = center_of_points([p.x for p in pop])
        score = 0
        for s in pair:
            # TODO: distance 和 fitness 归一化
            score += dist(s.x, center.x)
            score += s.fitness
        groups.append((pair, score))
    groups.sort(lambda x: x[1], reverse=True)
    return rank_based_select(groups, group_num)

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