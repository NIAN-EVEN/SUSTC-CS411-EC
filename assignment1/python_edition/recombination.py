from defination import *
from selection import *
import numpy as np
import copy, random

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

def one_bit_crossover(solution, _):
    x1 = solution[0].vec
    x2 = solution[1].vec
    yita1 = solution[0].yita
    yita2 = solution[1].yita
    x3 = copy.deepcopy(x2)
    x4 = copy.deepcopy(x1)
    yita3 = copy.deepcopy(yita2)
    yita4 = copy.deepcopy(yita1)
    bit = np.random.randint(0, len(solution)-1)
    for i in range(bit):
        x3[i] = x1[i]
        x4[i] = x2[i]
        yita3[i] = yita1[i]
        yita4[i] = yita2[i]
    return [Solution(x3, yita3), Solution(x4, yita4)]

##########################################################################
# #unused function
# def quadratic_recombination(solution, _):
#     # todo: 选择的父代相同时，会出现分母为0的现象
#     x1 = solution[0].vec
#     yita1 = solution[0].yita
#     fx1 = solution[0].fitness
#     x2 = solution[1].vec
#     yita2 = solution[1].yita
#     fx2 = solution[1].fitness
#     x3 = solution[2].vec
#     yita3 = solution[2].yita
#     fx3 = solution[2].fitness
#     fenzi = (x2**2-x3**2)*fx1+(x3**2-x1**2)*fx2+(x1**2-x2**2)*fx3
#     fenmu =   (x2 - x3) * fx1 + (x3 - x1) * fx2 + (x1 - x2) * fx3
#     x4 = (1/2) * (fenzi/fenmu)
#     yita4 = (yita1+yita2+yita3)/3
#     new_solution = Solution(x4, yita4)
#     if not new_solution.is_legal():
#         return [new_solution.correct()]
#     else:
#         return [new_solution]



