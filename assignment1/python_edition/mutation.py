import numpy as np
from defination import *
from scipy.stats import levy
import copy

def IFEP_mutation(solution):
    '''
    described in the cited paper
    :param solution: a ready-to-mutate solution
    :param func: benchmark function
    :return:
    '''
    cauchy_solution = cauchy_mutation(solution)
    gaussian_solution = gaussian_mutation(solution)
    if cauchy_solution.fitness >= gaussian_solution.fitness:
        return cauchy_solution
    else:
        return gaussian_solution

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

def levy_mutation(solution):
    x = solution.vec
    yita = solution.yita
    dim = x.size
    tao = 1 / np.sqrt(2 * np.sqrt(dim))
    tao_prime = 1 / np.sqrt(2 * dim)

    legal_offspring = False
    retry_time = 10
    while retry_time > 0:
        x_prime = x + yita * levy.rvs(size=dim)
        yita_prime = yita * np.exp(tao_prime * np.random.standard_normal() \
                                   + tao * np.random.standard_normal(dim))
        new_solution = Solution(x_prime, yita_prime)
        legal_offspring = new_solution.is_legal()
        retry_time = 0 if legal_offspring else retry_time - 1

    if not legal_offspring:
        new_solution.correct()

    return new_solution

def uniform_mutation(solution):
    # print("in uniform")
    x = solution.vec
    yita = solution.yita
    dim = x.size
    tao = 1 / np.sqrt(2 * np.sqrt(dim))
    tao_prime = 1 / np.sqrt(2 * dim)

    x_prime = x + yita * np.random.uniform(size=dim)
    yita_prime = yita * np.exp(tao_prime * np.random.standard_normal() \
                               + tao * np.random.standard_normal(dim))
    new_solution = Solution(x_prime, yita_prime)
    legal_offspring = new_solution.is_legal()

    if not legal_offspring:
        new_solution.correct()
    # print('after uniform')
    return new_solution

def one_bit_mutation(solution):
    # print("in one bit mutation")
    bit = np.random.randint(0, len(solution)-1)
    new_soluion = copy.deepcopy(solution)
    new_soluion.vec[bit] = np.random.uniform(Solution.lower_bound,
                                             Solution.upper_bound)
    # print('after one bit')
    return new_soluion
