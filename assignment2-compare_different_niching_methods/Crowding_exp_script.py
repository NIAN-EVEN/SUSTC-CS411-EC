import sys
sys.path.append("../")
from multiprocessing import Pool
from basic_definations_and_operators.Crossover import *
from basic_definations_and_operators.Mutation import *
from assignment2.cec2013.cec2013 import *
from assignment2.Crowding import *

def crowding_task(pop, crossover_idx, one_bit_mutate, param, func_idx,
                  mean_PR_table, mean_NSR_table, benchmark, crossover_operator,
                  accuracy_level, func, i):
    Solution.dimension = benchmark["dimension"]
    Solution.lower_bound = benchmark["lower_bound"]
    Solution.upper_bound = benchmark["upper_bound"]
    Solution.fitness_function = benchmark["fitness_function"]
    opt_cnt = crowding_update(pop=pop,
                              crossover=crossover_operator[crossover_idx],
                              mutate=one_bit_mutate,
                              param=param,
                              func_id=func_idx)
    X = np.array([p.x for p in pop])
    for accuracy_idx, accuracy in enumerate(accuracy_level):
        count, seeds = how_many_goptima(X, func, accuracy)
        # print("In the current population there exist", count, "global optimizers.")
        # print("Global optimizers: \n", seeds)
        mean_PR_table[accuracy_idx] += count
        if count == CEC2013(func_idx).get_no_goptima():
            mean_NSR_table[accuracy_idx] += 1
    print("%d-%d-%d" % (func_idx, crossover_idx, i))
    return opt_cnt, mean_PR_table, mean_NSR_table

def crowding():
    run = 50
    accuracy_filename = "crowding_accuracy_PR_SR.csv"
    opt_filename = "crowding_generation_optima_count.csv"
    function_idx = [1, 6, 11, 13]
    accuracy_level = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    mean_PR_table = np.zeros(len(accuracy_level))
    mean_NSR_table = np.zeros(len(accuracy_level))
    opt_record = []
    crossover_operator = [heuristic_crossover, arithmetic_crossover]
    for func_idx in function_idx:
        func = CEC2013(func_idx)
        param = {"pop_size": 500,
                 "evaluation_ub": func.get_maxfes(),
                 "crossover_alpha": 0.6}
        benchmark = {'fitness_function': func.evaluate,
                     'dimension': func.get_dimension(),
                     'lower_bound': func.get_lbound(0),
                     'upper_bound': func.get_ubound(0)}
        Solution.dimension = benchmark["dimension"]
        Solution.lower_bound = benchmark["lower_bound"]
        Solution.upper_bound = benchmark["upper_bound"]
        Solution.fitness_function = benchmark["fitness_function"]
        pop = init(param["pop_size"])
        for crossover_idx in range(len(crossover_operator)):
            result = []
            p = Pool()
            for i in range(run):
                result.append(p.apply_async(crowding_task, args=(pop, crossover_idx, one_bit_mutate,
                                                                 param, func_idx, mean_PR_table, mean_NSR_table,
                                                                 benchmark, crossover_operator, accuracy_level, func,
                                                                 i)))
            p.close()
            p.join()
            print("finished")
            for res in result:
                opt_cnt, PR_table, NSR_table = res.get()
                opt_record.append(opt_cnt)
                mean_PR_table += PR_table
                mean_NSR_table += NSR_table
            # opt_record
            msg = "f_%d_crossover_%d\n" % (func_idx, crossover_idx)
            msg += "generation, opt_num\n"
            for idx, record in enumerate(opt_record):
                msg = msg + str(idx) + ", " + str(record)[1:-1] + '\n'
            tofile(opt_filename, msg)
            opt_record = []
            # mean_PR_table
            # mean_NSR_table
            mean_PR_table = mean_PR_table / (run * CEC2013(func_idx).get_no_goptima())
            mean_NSR_table = mean_NSR_table / run
            msg = "f_%d_crossover_%d, PR, SR\n" % (func_idx, crossover_idx)
            for accuracy, PR, SR in zip(accuracy_level, mean_PR_table, mean_NSR_table):
                msg += "%.9f, %.3f, %.3f\n" % (accuracy, PR, SR)
            tofile(accuracy_filename, msg)
            print("write to file finished")
            mean_PR_table = np.zeros(len(accuracy_level))
            mean_NSR_table = np.zeros(len(accuracy_level))


if __name__ == "__main__":
    crowding()