import numpy as np

class Number():
    idx = 0
    def __init__(self, lowerBound, upperBound):
        self.num = np.random.randint(lowerBound, upperBound)
        self.idx = Number.idx
        Number.idx += 1

def encoding(p):
    p.gene = bin(p.num)[2:].format()
    if (p.gene.)
    pass

def decoding(p):
    pass

def func(x):
    return x**2

def evaluate(p):
    p.score = func(p)

def select(num, pop):
    seletedGroup = []
    # 轮盘赌方法
    sum = 0
    for p in pop:
        sum += p.score
    for i in range(num):
        randnum = np.random.rand() * sum
        seletedGroup.append(pop[int(randnum)])

    return seletedGroup

def crossover(k, p):
    # k-points crossover
    # 安全检查，基因长度是否大于k
    pos = np.random.randint() *
    pass

def mutation(p):
    pass

def multiplication(parents):
    pass

def elimination(offspring, pop):
    pass

def init(num, pop, lowerBound, upperBound):
    for i in range(num):
        pop.append(Number(lowerBound, upperBound))

if __name__ == "__main__":
    ##############################################
    pop = []
    popSize = 10
    gener = 100
    selectSize = 5
    lowerBound = 0
    upperBound = 31 + 1
    ##############################################
    init(popSize, pop, lowerBound, upperBound)
    for i in range(gener):
        # evaluate
        for j in range(popSize):
            evaluate(pop[j])
        # select
        parents = select(selectSize, pop)
        # generate offspring
        offspring = multiplication(parents)
        # eliminate
        pop = elimination(offspring, pop)
