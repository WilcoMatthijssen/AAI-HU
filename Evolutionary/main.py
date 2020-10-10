import random


def create_population(size):
    """ creates a random population by the size of size. """
    return [[random.randint(0,63) for _ in range(4)] for _ in range(size)]


def fitness(specimen):
    """ Get fitness of specimen. """
    A, B, C, D = specimen
    return pow((A-B), 2) + pow((C+D), 2) - pow((A-30), 3) - pow((C-40), 3)


def grade(populi):
    """ Gets average fitness of populi. """
    return sum([fitness(specimen) for specimen in populi]) / len(populi)


def evolve(populi, retain, random_select, mutate):
    """ Evolves the populi by the best performing ones with configurable randomness. """
    graded = [(fitness(specimen), specimen) for specimen in populi]
    graded = [x[1] for x in sorted(graded)]

    retain_size = int(len(graded)*retain)
    parents = graded[retain_size:]

    for specimen in graded[:retain_size]:
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

    parents += children
    return parents


if __name__ == '__main__':
    # Gekozen voor opdracht 6.2
    random.seed(0)

    populi = create_population(size=10_000)
    fitness_history = [grade(populi)]

    print("start", fitness_history[-1])

    for _ in range(25):
        populi = evolve(populi=populi, retain=0.5, random_select=0.0, mutate=0.5)
        score = grade(populi)
        fitness_history.append(score)
        #print(f"{score} is de gemiddelde score.")

    print("end", fitness_history[-1])

