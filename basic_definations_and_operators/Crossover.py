'''
crossover operator, interface is
1. population
2. crossover parameters
return a offspring list
'''
import sys
sys.path.append("../")
from basic_definations_and_operators.Selection import *
import numpy as np
import copy


def arithmetic_crossover(solution, alpha):
    x1 = solution[0].x
    x2 = solution[1].x
    x1_prime = alpha * x1 + (1-alpha) * x2
    x2_prime = alpha * x2 + (1-alpha) * x1
    return [Solution(x1_prime), Solution(x2_prime)]

def discrete_crossover(solution, _):
    # k point crossover
    k = np.random.rand()
    x1 = solution[0].x
    yita1 = solution[0].yita
    x2 = solution[1].x
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

def global_discrete_crossover(solution, _):
    x_new = np.zeros_like(solution[0].x)
    yita_new = np.zeros_like(solution[0].x)
    for i in range(Solution.dimension):
        parent = rank_based_select(solution, 1, None)
        x_new[i] = parent[0].x[i]
        yita_new[i] = parent[0].yita[i]
    return [Solution(x_new, yita_new)]

def one_bit_crossover(solution, _):
    x1 = solution[0].x
    x2 = solution[1].x
    x3 = copy.deepcopy(x2)
    x4 = copy.deepcopy(x1)
    bit = np.random.randint(0, len(solution)-1)
    for i in range(bit):
        x3[i] = x1[i]
        x4[i] = x2[i]
    return [Solution(x3), Solution(x4)]

def heuristic_crossover(solution, alpha):
    x1 = solution[0].x
    x2 = solution[1].x
    if solution[0].fitness > solution[1].fitness:
        direction = x1 - x2
    else:
        direction = x2 - x1
    new_solution1 = Solution(x1 + alpha * direction)
    new_solution2 = Solution(x2 + alpha * direction)
    return [new_solution1, new_solution2]

def quadratic_crossover(solution, _):
    # todo: 选择的父代相同时，会出现分母为0的现象
    x1 = solution[0].x
    fx1 = solution[0].fitness
    x2 = solution[1].x
    fx2 = solution[1].fitness
    x3 = solution[2].x
    fx3 = solution[2].fitness
    fenzi = (x2**2-x3**2)*fx1+(x3**2-x1**2)*fx2+(x1**2-x2**2)*fx3
    fenmu =   (x2 - x3) * fx1 + (x3 - x1) * fx2 + (x1 - x2) * fx3
    x4 = (1/2) * (fenzi/fenmu)
    new_solution = Solution(x4)
    new_solution.correct()
    return [new_solution]

##########################################################################
#unused function

