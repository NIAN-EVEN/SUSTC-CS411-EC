from basic_definations_and_operators.Solution import *
from basic_definations_and_operators.Crossover import *
from basic_definations_and_operators.Mutation import *
from assignment2_compare_different_niching_methods.cec2013.cec2013 import *
import random

def differential_evolution(pop, param, func_id, **kw):
    best_so_far = pop[0]
    opt_cnt = []
    evaluation_num = param["pop_size"]
    generation = 0
    while evaluation_num < param["evaluation_ub"]:
        for i, p in enumerate(pop):
            # print('evaluation=', evaluation_num)
            nn = nearest_neighbour(pop, p)
            r1, r2, r3, r4 = random.sample(pop, 4)
            while p in [r1, r2, r3, r4]:
                while p is r1:
                    r1 = random.sample(pop, 1)[0]
                while p is r2:
                    r2 = random.sample(pop, 1)[0]
                while p is r3:
                    r3 = random.sample(pop, 1)[0]
                while p is r4:
                    r4 = random.sample(pop, 1)[0]
            vp = DE_nrand_2(nn, r1, r2, r3, r4, param["F"]).x
            n = np.random.randint(Solution.dimension)
            L = 1
            while np.random.rand() < param["CR"] and L < Solution.dimension:
                L += 1
            vpp = np.zeros_like(vp)
            for j in range(Solution.dimension):
                if j >= n and j < n+L:
                    vpp[j] = vp[j]
                else:
                    vpp[j] = p.x[j]
            spp = Solution(vpp, generation)
            pop[i] = spp if spp.fitness > p.fitness else p
            evaluation_num += 1
        pop.sort(key=lambda x: x.fitness, reverse=True)
        # ################################## record #####################################
        # if generation % 10 == 0:
        #     X = np.array([p.x for p in pop])
        #     count, seeds = how_many_goptima(X, CEC2013(func_id), 0.0001)
        #     opt_cnt.append(count)
        # ################################## record #####################################
        generation += 1
        ################################# test ########################################
        # if compare(pop[0].real_fitness, best_so_far.real_fitness) or generation % 50 == 0:
        #     if compare(pop[0].real_fitness, best_so_far.real_fitness):
        #         best_so_far = pop[0]
        #     print("evaluation=%d, %f" % (evaluation_num, best_so_far.real_fitness))
        if generation % 50 == 0:
            X = np.array([p.x for p in pop])
            count, seeds = how_many_goptima(X, CEC2013(func_id), 0.0001)
            print("accuracy: %f, count: %d" % (0.0001, count))
        ################################# test ########################################
    X = np.array([p.x for p in pop])
    count, seeds = how_many_goptima(X, CEC2013(func_id), 0.0001)
    opt_cnt.append(count)
    return opt_cnt

def differential_evolution_niche(pop, param, func_id, **kw):
    best_so_far = pop[0]
    opt_cnt = []
    evaluation_num = param["pop_size"]
    generation = 0
    while evaluation_num < param["evaluation_ub"]:
        for i, p in enumerate(pop):
            # print('evaluation=', evaluation_num)
            nn = nearest_neighbour(pop, p)
            r1, r2, r3, r4 = random.sample(pop, 4)
            while p in [r1, r2, r3, r4]:
                while p is r1:
                    r1 = random.sample(pop, 1)[0]
                while p is r2:
                    r2 = random.sample(pop, 1)[0]
                while p is r3:
                    r3 = random.sample(pop, 1)[0]
                while p is r4:
                    r4 = random.sample(pop, 1)[0]
            vp = DE_nrand_2(nn, r1, r2, r3, r4, param["F"]).x
            n = np.random.randint(Solution.dimension)
            L = 1
            while np.random.rand() < param["CR"] and L < Solution.dimension:
                L += 1
            vpp = np.zeros_like(vp)
            for j in range(Solution.dimension):
                if j >= n and j < n+L:
                    vpp[j] = vp[j]
                else:
                    vpp[j] = p.x[j]
            spp = Solution(vpp, generation)
            niche = CEC2013(func_id).get_rho()
            pop[i] = spp if spp.fitness > p.fitness and distance(spp, p) > niche else p
            evaluation_num += 1
        pop.sort(key=lambda x: x.fitness, reverse=True)
        # ################################## record #####################################
        # if generation % 10 == 0:
        #     X = np.array([p.x for p in pop])
        #     count, seeds = how_many_goptima(X, CEC2013(func_id), 0.0001)
        #     opt_cnt.append(count)
        # ################################## record #####################################
        generation += 1
        ################################# test ########################################
        # if compare(pop[0].real_fitness, best_so_far.real_fitness) or generation % 50 == 0:
        #     if compare(pop[0].real_fitness, best_so_far.real_fitness):
        #         best_so_far = pop[0]
        #     print("evaluation=%d, %f" % (evaluation_num, best_so_far.real_fitness))
        if generation % 50 == 0:
            X = np.array([p.x for p in pop])
            count, seeds = how_many_goptima(X, CEC2013(func_id), 0.0001)
            print("accuracy: %f, count: %d" % (0.0001, count))
        ################################# test ########################################
    X = np.array([p.x for p in pop])
    count, seeds = how_many_goptima(X, CEC2013(func_id), 0.0001)
    opt_cnt.append(count)
    return opt_cnt

def differential_evolution_update(pop, param, func_id=None, **kw):
    best_so_far = pop[0]
    opt_cnt = []
    evaluation_num = param["pop_size"]
    generation = 0
    while evaluation_num < param["evaluation_ub"]:
        for i, p in enumerate(pop):
            # print('evaluation=', evaluation_num)
            r1, r2 = tripling(pop, p)
            spp = quadratic_crossover([p, r1, r2], None)[0]
            pop[i] = spp if spp.fitness > p.fitness else p
            evaluation_num += 1
        pop.sort(key=lambda x: x.fitness, reverse=True)
        # if generation % 10 == 0:
        #     X = np.array([p.x for p in pop])
        #     count, seeds = how_many_goptima(X, CEC2013(func_id), 0.0001)
        #     opt_cnt.append(count)
        if compare(pop[0].fitness, best_so_far.fitness) or generation % 50 == 0:
            best_so_far = pop[0]
            print("evaluation=%d, %s" % (evaluation_num, pop[0]))
        if pop[-1].fitness == pop[0].fitness:
            break
        generation += 1
    # X = np.array([p.x for p in pop])
    # count, seeds = how_many_goptima(X, CEC2013(func_id), 0.0001)
    # opt_cnt.append(count)
    # return opt_cnt

def tripling(pop, p):
    nn1 = pop[0]
    nn1_idx = 0
    min_distance1 = distance(p, nn1)
    for i, q in enumerate(pop):
        dist = distance(q, p)
        if dist < min_distance1:
            nn1 = q
            nn1_idx = i
            min_distance1 = dist
    nn2 = pop[0]
    min_distance2 = distance(p, nn2)
    while nn2 is nn1:
        nn2 = random.sample(pop, 1)[0]
        min_distance2 = distance(p, nn2)
    for i, q in enumerate(pop):
        dist = distance(q, p)
        if dist < min_distance2 and i != nn1_idx:
            nn2 = q
            nn2_idx = i
            min_distance2 = dist
    return nn1, nn2

def DE_nrand_2(nn, r1, r2, r3, r4, F):
    sx = nn.x + F * (r1.x - r2.x) + F * (r3.x - r4.x)
    return Solution(sx)

if __name__ == "__main__":
    # function_idx = [1, 6, 11, 13]
    function_idx = [11, 13]
    accuracy_level = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    result = []
    for func_idx in function_idx:
        func = CEC2013(func_idx)
        param = {"pop_size": 100,
                 "evaluation_ub": func.get_maxfes(),
                 "F": 0.5,
                 "CR": 0.4}
        Solution.dimension = func.get_dimension()
        Solution.lower_bound = func.get_lbound(0)
        Solution.upper_bound = func.get_ubound(0)
        Solution.fitness_function = func.evaluate
        pop = init(param["pop_size"])
        differential_evolution(pop=pop,
                               param=param,
                               func_id=func_idx)
        X = np.array([p.x for p in pop])
        for accuracy in accuracy_level:
            count, seeds = how_many_goptima(X, func, accuracy)
            print("accuracy: %f, global count: %d" % (accuracy, count))
            if accuracy == 0.00001:
                result.append(count)
    for func_idx, opt_cnt, accuracy in zip(function_idx, result, accuracy_level):
        print("function:%d, opt_cnt:%d, real_opt:%d, accuracy:%f" %
              (func_idx, opt_cnt, CEC2013(func_idx).get_no_goptima(), accuracy))