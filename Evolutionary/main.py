import random
import numpy as np


def get_population(size):
    return [[random.randint(0,63) for _ in range(4)] for _ in range(size)]


def fitness(specimen, target):
    A, B, C, D = specimen
    lift = pow((A-B), 2) + pow((C+D), 2) - pow((A-30), 3) - pow((C-40), 3)
    return abs(target - lift)


def grade(populi, target):
    # print(populi)
    return np.average([fitness(specimen, target) for specimen in populi])


def evolve(populi, target, retain, random_select, mutate):
    graded = [(fitness(specimen, target), specimen) for specimen in populi]
    graded = [x[1] for x in sorted(graded)]

    retain_size = int(len(graded)*retain)
    parents = graded[:retain_size]

    for specimen in graded[retain_size:]:
        if random_select > random.random():
            parents.append(specimen)

    children = []
    for _ in range(len(populi)-len(parents)):
        male, female = random.sample(parents, 2)
        children.append(male[:2] + female[2:])

    for child in children:
        if mutate > random.random():
            mutated_pos = random.randint(0, len(child)-1)
            child[mutated_pos] = random.randint(min(child), max(child))

    parents.extend(children)
    return parents


if __name__ == '__main__':
    # Gekozen voor opdracht 6.2

    random.seed(0)

    populi = get_population(1000)
    target = 63*4
    fitness_history = [grade(populi, target)]

    for _ in range(10):
        populi = evolve(populi=populi, target=target, retain=0.1, random_select=0.05, mutate=0.1)
        score = grade(populi, target)
        fitness_history.append(score)
        print(f"Gemiddeld {score} van het target af.")

