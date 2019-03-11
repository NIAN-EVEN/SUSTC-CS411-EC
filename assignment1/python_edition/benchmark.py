import numpy as np

# dimension 30, -32<x<32
def ackley_function(target_vector):
    # target_vector = target_vector.vector
    length = len(target_vector)
    return -20 * np.exp(-0.2 * np.sqrt(sum(target_vector ** 2) / length)) - np.exp(sum(np.cos(2 * np.pi * target_vector)) / length) + 20 + np.exp(1)


# dimension 2, -2<x<2
def gold_stein_price_function(target):
    # target = x.vector
    x_one = target[0]
    x_two = target[1]
    return (1 + ((x_one + x_two + 1) ** 2) * (19 - 14 * x_one + 3 * (x_one ** 2) - 14 * x_two + 6 * x_one * x_two + 3 * x_two * x_two)) * \
           (30 + ((2 * x_one - 3 * x_two) ** 2) * (18 - 32 * x_one + 12 * (x_one ** 2) + 48 * x_two - 36 * x_one * x_two + 27 * x_two * x_two))


# dimension 30, -1.28<x<1.28
def quartic_noise_function(target_vector):
    # target_vector = target_vector.vector
    sequence_vector = np.array([i + 1 for i in range(30)])
    return np.dot(sequence_vector, target_vector ** 4) + np.random.random()


# dimension 30, -5.12<x<5.12
def rastrigin_function(target_vector):
    # target_vector = target_vector.vector
    return sum(target_vector ** 2 - 10 * np.cos(2 * np.pi * target_vector) + np.array([10 for i in range(30)]))


# dimension 30, -100<x<100
def step_function(target_vector):
    # target_vector = target_vector.vector
    return sum(np.floor(target_vector + np.array([.5 for i in range(30)])) ** 2)


# dimension 4, 0<x<10
def shekel_function(target_vector):
    # target_vector = target_vector.vector
    y = 0
    c = [0.1, 0.2, 0.2, 0.4, 0.4]
    a = [[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7]]
    for i in range(len(c)):
        base = c[i]
        for j in range(4):
            base += (target_vector[j] - a[i][j]) ** 2
        y += 1 / base
    return -y


# dimension 4, -100<x<100
def flat_function(target_vector):
    target_vector = target_vector.vector
    for item in target_vector:
        if item < -1 or item > 1:
            return 1
    return 0


def easom_function(target_vector):
    target_vector = target_vector.vector
    x_one = target_vector[0]
    x_two = target_vector[1]
    return -np.cos(x_one) * np.cos(x_two) * np.exp(- ((x_one - np.pi) ** 2 + (x_two - np.pi) ** 2))


benchmark_lst = [(step_function, 30, -100, 100), (quartic_noise_function, 30, -1.28, 1.28), (rastrigin_function, 30, -5.12, 5.12), (ackley_function, 30, -32, 32), (gold_stein_price_function, 2, -2, 2),  (shekel_function, 4, 0, 10)]
generation_lst = [1400, 2000, 5000, 5000, 20000, 1400, 3000]

if __name__ == "__main__":
    print(np.sqrt([1, 2, 3]))
    print(sum(np.array([1, 2, 3])))
    print(np.exp(1))
    # print(np.random.random(size=30))
    # print(step_function(np.array([0 for i in range(30)])))
    # print(easom_function(np.array([2, 3])))

