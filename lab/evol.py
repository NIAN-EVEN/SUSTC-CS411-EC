import random
import collections


def evolutionary_framework_local(generation, population, generator, sorter, fitness,
                                 acceptable, selector, mutation, crossover):
    population_lst = generator(population)
    population_lst = [(item, fitness(item)) for item in population_lst]
    population_lst = sorter(population_lst)
    for i in range(generation):
        child_lst = list()
        for single_item in population_lst:
            if selector(single_item, population_lst):
                if mutation is not None:
                    result_item = mutation(single_item)
                    if isinstance(result_item, collections.Iterable):
                        child_lst += result_item
                    else:
                        child_lst.append(result_item)
                if crossover is not None:
                    child_lst.append(crossover(single_item, population_lst))
        child_lst = [(item, fitness(item)) for item in child_lst]
        population_lst += child_lst
        population_lst = sorter(population_lst)
        # if population_lst[0][1] == population_lst[-1][1]:
        #     np.random.shuffle(population_lst)
        population_lst = population_lst[: population]
        print("In " + str(i) + " generation: ")
        print(population_lst)
        if acceptable is not None and acceptable(population_lst):
            return population_lst, i
    return population_lst, generation


def roulette_wheel_selection(single_item, population_lst):
    select_prop = single_item[1] / sum([item[1] for item in population_lst])
    current_prop = random.random()
    return True if select_prop <= current_prop else False


def binary_to_integer(binary_str):
    # print("testing: " , binary_str)
    return int(binary_str, 2)


def integer_to_binary(target):
    target = bin(target).replace("0b", "")
    return "".join(['0' for i in range(len(bin(31)) - 2 - len(target))]) + target


def mutation_factory(mutation_rate):
    def _mutation(single_item):
        real_item = single_item[0]
        binary_item = integer_to_binary(real_item)
        binary_lst = list()
        for single_char in binary_item:
            current_prop = random.random()
            if current_prop <= mutation_rate:
                binary_lst.append('1' if single_char == '0' else '0')
            else:
                binary_lst.append(single_char)
        return binary_to_integer("".join(binary_lst))

    return _mutation


def crossover_factory(crossover_rate):
    def _crossover(single_item, population_lst):
        real_item = single_item[0]
        binary_item = integer_to_binary(real_item)
        total_score = sum([item[1] for item in population_lst])
        prop_lst = [item[1] / total_score for item in population_lst]
        real_prop_lst = list()
        tmp_prop = 0
        for item in prop_lst:
            real_prop_lst.append(item + tmp_prop)
            tmp_prop += item
        prop_value = random.random()
        real_index = len(real_prop_lst) - 1
        for i in range(len(real_prop_lst)):
            if real_prop_lst[i] < prop_value:
                real_index = i
                break
        other_item = population_lst[real_index][0]
        other_binary = integer_to_binary(other_item)
        result_lst = list()
        for i in range(len(other_binary)):
            if random.random() < .5:
                result_lst.append(binary_item[i])
            else:
                result_lst.append(other_binary[i])
        return binary_to_integer("".join(result_lst))

    return _crossover


def fitness(item):
    return item * item


def sorter(population_lst):
    population_lst.sort(key=lambda x: x[1])
    population_lst.reverse()
    return population_lst


if __name__ == "__main__":
    mutation = mutation_factory(.5)
    crossover = crossover_factory(.5)
    population = 10
    generation = 10
    result_lst = evolutionary_framework_local(generation, population,
                                              lambda x: [random.randint(0, 31) for i in range(x)], sorter, fitness,
                                              None, roulette_wheel_selection, mutation, crossover)
    print(result_lst)