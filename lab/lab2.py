import numpy as np
import random

class Vector():
    idx = 0
    def __init__(self, lowerBound=-2, upperBound=2, dimension=2):
        # TODO: random interval here is [lowerBound, upperBound)，but we need [low, up]
        self.num = np.random.uniform(lowerBound, upperBound, dimension)
        self.idx = Vector.idx
        Vector.idx += 1

    def setNumber(self, vector):
        self.num = vector

def func(x):
    if len(x) != 2:
        print("error occured in func()")
        exit(-1)
    a = 1+((x[0]+x[1]+1)**2)(19-14*x[0])
    return a

def evaluate(p):
    p.score = func(p.num)

def rankBasedSelect(num, pop):
    '''在pop中选择num个个体'''
    pop.sort(key=lambda x:x.score)
    range = len(pop) * (len(pop) + 1) / 2
    # rand为0-range的随机数
    rand = np.random.uniform(0, range, num)
    # 在整个range中j占比∑(1~j-1)-∑(1~j)部分
    # 对rand反向求是哪个数累加而成再加1即实现按照排序选择的功能x
    reverseParentsPos = (np.sqrt(8 * rand + 1) - 1) / 2 + 1
    parents = []
    for reverseParent in reverseParentsPos.tolist():
        parents.append(pop[len(pop) - int(reverseParent)])
    return parents

def muLambdaTrunction(lamb=4):
    pass

def globalDiscreteRecombination(parents, offNum):
    offs = []
    for i in range(offNum):
        newGene = np.zeros(parents[0].dim)
        for j in range(parents[0].dim):
            pos = random.randint(0, len(parents))
            newGene[j] = parents[pos].num[j]
        newOffs = Vector()
        newOffs.setNumber(newGene)
        offs.append(newOffs)
    return offs


def quadraticApproxination(offsprings):
    pass

def mutation(p, muSize):
    pos = []
    for i in range(muSize):
        tmp = np.random.randint(np.log2(upperBound))
        if tmp not in pos:
            pos.append(tmp)
            if p.gene[tmp] == '0':
                p.gene[tmp] = '1'
            elif p.gene[tmp] == '1':
                p.gene[tmp] = '0'
    p.setNumber(p.gene)

def reproduce(parents, k):
    pass

def printInfo(round, pop):
    print("round {0}: ".format(round))
    for p in pop:
        print("idx: {0}, value: {1}, score: {2}".format(p.idx, p.num, p.score))

def init(num, pop, lowerBound, upperBound, dimen):
    for i in range(num):
        pop.append(Vector(lowerBound, upperBound, dimen))
    # evaluate
    for i in range(popSize):
        evaluate(pop[i])

if __name__ == "__main__":
    ##############################################
    pop = []
    popSize = 50
    gener = 500000/50
    selectSize = 5
    lowerBound = -2
    upperBound = 2
    k = 2
    muSize = 1
    muRate = 0.1
    dimension = 2
    ##############################################
    init(popSize, pop, lowerBound, upperBound, dimension)
    for i in range(gener):
        # select
        parents = rankBasedSelect(selectSize, pop)
        # generate offspring
        offs = reproduce(parents, k)
        for off in offs:
            mutation(off, muSize)
        pop.append(offs)
        # eliminate
        muLambdaTrunction(pop)
        printInfo(i, pop)

