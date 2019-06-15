from basic_definations_and_operators.Solution import *
from itertools import combinations


def speciation_select(pop, select, num, sigma_mate, iter_num=1000):
    '''
    根据距离判断num个个体是否在同一个组
    :param pop:
    :param select: 选择函数
    :param num: 每次选择的组数
    :param sigma_mate: 同组半径
    :param iter_num: 最大迭代次数
    :return:
    '''
    # todo: 没有距离点时死循环
    distance_table = distance_matrix(pop)
    select_pop = []
    select_num = 0
    while len(select_pop) < num and select_num < iter_num:
        inAGroup = True
        offspring = select(pop, num)
        for off1, off2 in combinations(offspring, 2):
            if distance(off1, off2) > sigma_mate:
                inAGroup = False
                break
        if inAGroup == True:
            select_pop.extend(offspring)
        select_num += 1
    if len(select_pop) < num:
        select_pop.extend(select(pop, num-len(select_pop)))
    return select_pop
