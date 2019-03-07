import numpy as np

class Solution():
    def __init__(self, x, yita, score):
        self.vec = x
        self.score = score
        self.yita = yita

    def __len__(self):
        return self.vec.size

def optimise(benchmark, budget, recombination, mutation, selection):
    # get configuration
    func = benchmark[0]
    dimension = benchmark[1]
    lower_bound = benchmark[2]
    upper_bound = benchmark[3]
    # init population

def fitness(func, x):
    objValue = func(x)
    return objValue

def init(num, dim, ub, lb, func):
    pop = []
    for i in range(num):
        x = np.random.rand(dim)*(ub - lb) + ub
        yita = np.random.rand(dim)
        pop.append(Solution(x, yita, fitness(func, x)))
    pop.sort(key=lambda x:x.score)

def recombination():
    pass

def FEP_mutation(x, yita):
    dim = x.size
    cauchy_j = np.random.standard_cauchy(dim)
    gaussian = np.random.standard_normal(dim)
    tao = 1/np.sqrt(2*np.sqrt(dim))
    tao_prime = 1/np.sqrt(2*np.sqrt(dim))

    x_prime = x+yita*gaussian
    yita_prime = yita*np.exp(tao_prime*gaussian+tao*cauchy_j)

    return x_prime, yita_prime

def CEP_mutation(x, yita):
    dim = x.size
    gaussian_j = np.random.standard_normal(dim)
    gaussian = np.random.standard_normal(dim)
    tao = 1/np.sqrt(2*np.sqrt(dim))
    tao_prime = 1/np.sqrt(2*np.sqrt(dim))

    x_prime = x+yita*gaussian_j
    yita_prime = yita*np.exp(tao_prime*gaussian+tao*gaussian_j)

    return x_prime, yita_prime




if __name__ == "__main__":
