import numpy as np

class Number():
    idx = 0
    def __init__(self, lowerBound, upperBound):
        self.num = np.random.randint(lowerBound, upperBound)
        self.idx = Number.idx
        Number.idx += 1

    def setNumber(self, gene):
        binary = ""
        for g in gene:
            binary += g
        self.num = int(binary, 2)

def encoding(p):
    gene = bin(p.num)[2:].rjust(int(np.log2(upperBound)), '0')
    p.gene = [x for x in gene]

def decoding(p):
    gene = ""
    for g in p.gene:
        gene += g
    p.num = int(p.gene, 2)

def func(x):
    return x**2

def evaluate(p):
    p.score = func(p.num)

def select(num, pop):
    seletedGroup = []
    # 轮盘赌方法
    sum = 0
    for p in pop:
        sum += p.score
    for i in range(num):
        randnum = np.random.rand() * sum
        number = 0
        for p in pop:
            if number <= randnum and number + p.score > randnum:
                seletedGroup.append(p)
                break
            number += p.score

    return seletedGroup

def crossover(p1, p2, k):
    # k-points crossover
    encoding(p1)
    encoding(p2)
    # 安全检查，基因长度是否大于k
    if len(p1.gene) <= k or len(p2.gene) <= k:
        print("gene length is {0}, {1}, k is {2}".format(len(p1.gene), len(p2.gene), k))
        exit(-1)
    # 产生k个不同的随机数
    pos = []
    for i in range(k):
        tmp = np.random.randint(len(p1.gene))
        if tmp not in pos:
            pos.append(tmp)
    newGene1 = ""
    newGene2 = ""
    flag = True
    for i in range(len(p1.gene)):
        if i in pos:
            flag = not flag
        if flag:
            newGene1 += p1.gene[i]
            newGene2 += p2.gene[i]
        else:
            newGene1 += p2.gene[i]
            newGene2 += p1.gene[i]
    offspring1 = Number(lowerBound, upperBound)
    offspring1.setNumber(newGene1)
    evaluate(offspring1)
    offspring2 = Number(lowerBound, upperBound)
    offspring2.setNumber(newGene1)
    evaluate(offspring2)
    return offspring1, offspring2

def mutation(p, muSize):
    encoding(p)
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

def multiplication(parents, k):
    return crossover(parents[0], parents[1], k)

def elimination(pop):
    pop.sort(key=lambda x: x.score, reverse=True)
    while len(pop) > popSize:
        pop.pop(-1)

def init(num, pop, lowerBound, upperBound):
    for i in range(num):
        pop.append(Number(lowerBound, upperBound))

def printInfo(round, pop):
    print("round {0}: ".format(round))
    for p in pop:
        print("idx: {0}, value: {1}, score: {2}".format(p.idx, p.num, p.score))

if __name__ == "__main__":
    ##############################################
    pop = []
    popSize = 10
    gener = 100
    selectSize = 5
    lowerBound = 0
    upperBound = 31 + 1
    k = 2
    muSize = 1
    muRate = 0.1
    ##############################################
    init(popSize, pop, lowerBound, upperBound)
    for i in range(gener):
        # evaluate
        for j in range(popSize):
            evaluate(pop[j])
        for j in range(selectSize):
            # select
            parents = select(2, pop[0:popSize])
            # generate offspring
            for offs in multiplication(parents, k):
                if np.random.rand() < muRate:
                    mutation(offs, muSize)
                pop.append(offs)
        # eliminate
        elimination(pop)
        printInfo(i, pop)

