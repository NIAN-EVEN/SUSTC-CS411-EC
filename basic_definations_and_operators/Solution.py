from math import isnan
import numpy as np

class Solution():
    dimension = None
    lower_bound = None
    upper_bound = None
    fitness_function = None
    def __init__(self, x, generation=None, yita=None):
        self.x = x
        self.correct()
        self.yita = yita
        if Solution.fitness_function != None:
            self.fitness = Solution.fitness_function(x)
            self.real_fitness = self.fitness
        else:
            self.fitness = None
            self.real_fitness = None
        self.generation = generation

    def is_legal(self):
        for val in self.x:
            if isnan(val) or val > Solution.upper_bound \
                or val < Solution.lower_bound:
                return False
        # print("yes legal")
        return True

    def correct(self):
        for i, val in enumerate(self.x):
            if isnan(val):
                self.x[i] = np.random.uniform(Solution.lower_bound,
                                                Solution.upper_bound)
            elif val > Solution.upper_bound:
                self.x[i] = Solution.upper_bound
            elif  val < Solution.lower_bound:
                self.x[i] = Solution.lower_bound

    def __len__(self):
        return self.x.size

    def __str__(self):
        return "generation=" + str(self.generation) + ", fitness=" + str(self.fitness) + ", x=" + str(self.x)

def fitness(func, x):
    objValue = func(x)
    return objValue

def tofile(filename, msg):
    with open(filename, "a") as f:
        f.write(msg)

def init(num):
    pop = []
    for i in range(num):
        x = np.random.uniform(Solution.lower_bound,
                              Solution.upper_bound,
                              size=Solution.dimension)
        pop.append(Solution(x, 0))
    pop.sort(key=lambda x: x.fitness)
    return pop

def uniform_init(num):
    pop = []
    for i in range(num):
        # todo: 点在空间均匀分布
        pass

def refit(pop):
    for p in pop:
        p.fitness = Solution.fitness_function(p.x)
        p.real_fitness = p.fitness

def compare(a, b):
    return a > b

def distance_matrix(pop):
    pop_size = len(pop)
    distance_table = np.zeros((pop_size, pop_size))
    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            distance_table[i, j] = distance(pop[i], pop[j])
            distance_table[j, i] = distance_table[i, j]
    return distance_table

def distance(p1, p2):
    vector_length = len(p1)
    dist = 0
    for i in range(vector_length):
        dist += (p1.x[i] - p2.x[i]) ** 2
    return np.sqrt(dist)

def nearest_neighbour(pop, p):
    min_distance = distance(p, pop[0])
    nn = pop[0]
    for q in pop:
        dist = distance(q, p)
        if dist < min_distance:
            nn = q
            min_distance = dist
    return nn

def nearest_pairing(pop):
    '''
    :param pop: population
    :return: father group and mother group index
    '''
    if len(pop) % 2 != 0:
        print("pop size = %d, it's a odd number, cannot paring...")
        exit(-1)
    distance_table = distance_matrix(pop)
    occupy = np.zeros(len(pop))
    father_idx = []
    mather_idx = []
    while not occupy.all():
        # 选一个没有被occupy的点放进father
        for idx in range(len(occupy)):
            if occupy[idx] == 0:
                father_idx.append(idx)
                occupy[idx] = 1
                break
        fidx = father_idx[-1]
        # 选一个距离father最近的没有被occupy的点放进mather
        midx = distance_table[fidx].argmax()
        dist = distance_table[fidx].max()
        for idx in range(len(occupy)):
            if occupy[idx] == 0 and distance_table[fidx, idx] <= dist:
                dist = distance_table[father_idx[-1], idx]
                midx = idx
        mather_idx.append(midx)
        occupy[midx] = 1
    return father_idx, mather_idx