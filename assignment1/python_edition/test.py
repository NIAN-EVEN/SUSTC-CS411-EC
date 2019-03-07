from benchmark import benchmark_lst, generation_lst


def test_for_first_submission():
    arithmetic_recombination = arithmetic_recombination_factory(.3)
    population = 50
    offsprings = 50
    rounds = 10
    for index in range(len(benchmark_lst)):
        functions = benchmark_lst[index]
        generation = 50  # using previous statement to set generation to config
        # generation = generation_lst[index]
        fitness, dimension, low, up = functions
        Solution.has_bounded = True
        Solution.dimension = dimension
        Solution.low = low
        Solution.up = up
        print("=====================================================================")
        print("Testing " + str(fitness))
        score_lst = list()
        for i in range(rounds):
            result_lst, generation = evolutionary_framework_local(generation, population, generator,
                                                                  rank_population_selector,
                                                                  offsprings, fitness, None, rank_parent_selector,
                                                                  cauchy_mutation_for_multiple_item, arithmetic_recombination)
            result_lst.sort(key=lambda x: x[1])
            score_lst.append((result_lst[0][1], result_lst[0][0].vector))
            print("Round " + str(i) + " is finished.")
        score_lst.sort(key=lambda x: x[0])
        print("Best value is: " + str(score_lst[0][0]))
        print("Best solution is: " + str(score_lst[0][1]))
        print("Average value is: " + str(np.mean([item[0] for item in score_lst])))
        print("Var is " + str(np.var([item[0] for item in score_lst])))
        print("=====================================================================")