from math import isnan
import numpy as np

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
