import sys
sys.path.append("../")
from basic_definations_and_operators.Solution import *
from scipy.stats import levy
import copy

def IFEP_mutate(solution):
    '''
    described in the cited paper
    :param solution: a ready-to-mutate solution
    :param func: benchmark function
    :return:
    '''
    cauchy_solution = cauchy_mutate(solution)
    gaussian_solution = gaussian_mutate(solution)
    if cauchy_solution.fitness >= gaussian_solution.fitness:
        return cauchy_solution
    else:
        return gaussian_solution

def cauchy_mutate(solution):
    # print("in cauchy")
    check_yita(solution)
    x = solution.x
    yita = solution.yita
    dim = x.size
    tao = 1 / np.sqrt(2 * np.sqrt(dim))
    tao_prime = 1 / np.sqrt(2 * dim)
    # print("real in cauchy")
    x_prime = x + yita*np.random.standard_cauchy(dim)
    yita_prime = yita*np.exp(tao_prime*np.random.standard_normal() \
                             +tao*np.random.standard_normal(dim))
    n_solution = Solution(x_prime, yita=yita_prime)
    return n_solution

def gaussian_mutate(solution):
    # print("in gaussian")
    check_yita(solution)
    x = solution.x
    yita = solution.yita
    dim = x.size
    tao = 1 / np.sqrt(2 * np.sqrt(dim))
    tao_prime = 1 / np.sqrt(2 * dim)

    x_prime = x+yita*np.random.standard_normal(dim)
    yita_prime = yita*np.exp(tao_prime*np.random.standard_normal() \
                             + tao*np.random.standard_normal(dim))
    new_solution = Solution(x_prime, yita=yita_prime)
    return new_solution

def levy_mutate(solution):
    # todo: disable yita and .correct
    x = solution.x
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

def uniform_mutate(solution):
    # todo: disable yita
    x = solution.x
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

def one_bit_mutate(solution):
    # print("in one bit mutate")
    bit = np.random.randint(0, len(solution))
    new_soluion = copy.deepcopy(solution)
    new_soluion.x[bit] = np.random.uniform(Solution.lower_bound,
                                             Solution.upper_bound)
    refit([new_soluion])
    return new_soluion

def check_yita(solution):
    if solution.yita == None:
        solution.yita = np.zeros(Solution.dimension) + 3