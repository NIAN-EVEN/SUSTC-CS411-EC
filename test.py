from multiprocessing import Pool
from copy import deepcopy

def process1(pop):
    print("process1")
    pop[0][0] = 'process1'
    return pop

def process2(pop):
    print("process2")
    pop[0][0] = 'process2'
    return pop

def process3(pop):
    print("process3")
    pop[0][0] = 'process3'
    return pop

if __name__ == "__main__":
    p = Pool(3)
    a = [[1,2,3], [4,5,6], [7,8,9]]
    result = []
    print("sub process start:")
    result.append(p.apply_async(process1, args=(a,)))
    result.append(p.apply_async(process2, args=(a,)))
    result.append(p.apply_async(process3, args=(a,)))
    p.close()
    p.join()
    print(a)
    print("sub process finish:")
    for i in result:
        print(i.get())

